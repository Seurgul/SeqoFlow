"""Continuous Poisson workloads on a single, step-preemptible simulated worker.

All times are virtual seconds: host sleeps and logging cannot change offered load.
The measurement cohort is selected by arrival time, never by completion order.
"""

import argparse
from dataclasses import dataclass
import heapq
import itertools
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import statistics


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def service_times(traces, include_profiling=False):
    if not traces or any(not trace for trace in traces):
        raise ValueError("Provide at least one trace; traces must contain steps")
    durations = []
    for trace in traces:
        steps = []
        for step in trace:
            fields = ("exec", "prf") if include_profiling else ("exec",)
            times = [float(step.get(field, 0.0)) for field in fields]
            estimate = float(step.get("est_rm", 0.0))
            if not all(math.isfinite(t) and t >= 0 for t in times):
                raise ValueError("Service times must be finite and nonnegative")
            if not math.isfinite(estimate):
                raise ValueError("Remaining-time estimates must be finite")
            steps.append(sum(times))
        durations.append(steps)
    if statistics.mean(map(sum, durations)) <= 0:
        raise ValueError("Mean trace service time must be positive")
    return durations


@dataclass
class Job:
    qid: int
    trace_id: int
    arrival: float
    measured: bool
    high: bool
    step: int = 0
    served: float = 0.0
    level: int = 0
    step_remaining: float = 0.0
    quantum_used: float = 0.0
    pdb_start: float = None
    pdb_high_service: float = 0.0
    smoothed_remaining: float = None
    estimate_step: int = -1
    estimate_served: float = 0.0


class Policy:
    """Scheduling policy for a single virtual-time worker."""

    def __init__(self, name, traces, durations, alpha, mlfq_levels=4,
                 mlfq_base_quantum=None, mlfq_boost_interval=None,
                 pdb_batch_size=4, pdb_starvation_threshold=0.75,
                 pdb_admission_interval=None, srtf_smoothing=0.2):
        self.name = name
        self.traces = traces
        totals = list(map(sum, durations))
        self.alpha = alpha
        if not math.isfinite(srtf_smoothing) or not 0 < srtf_smoothing <= 1:
            raise ValueError("SRTF smoothing must be in (0, 1]")
        self.srtf_smoothing = srtf_smoothing
        self.ready = []
        self.ready_heap = []
        self.ready_order = itertools.count()
        self.paused = None
        self.high_pending = []
        self.high_batch = []
        self.low_ready = []
        self.pdb_batch_size = pdb_batch_size
        self.pdb_starvation_threshold = pdb_starvation_threshold
        positive_steps = [step for trace in durations for step in trace if step > 0]
        base = mlfq_base_quantum or statistics.median(positive_steps)
        self.quanta = [base * (2 ** level) for level in range(mlfq_levels)]
        self.boost_interval = mlfq_boost_interval or 20 * statistics.mean(totals)
        self.pdb_admission_interval = (pdb_admission_interval or
                                       statistics.median(positive_steps))

    def add(self, job, front=False):
        if self.name == "fcfs":
            heapq.heappush(self.ready_heap, (job.qid, job))
            return
        if self.name == "srtf":
            # Every waiting job's aging score contains the same -alpha*now.
            # Remove that common term so the heap key stays fixed while it waits.
            priority = (self.remaining_estimate(job)
                        + self.alpha * (job.arrival + job.served))
            heapq.heappush(self.ready_heap,
                           (priority, next(self.ready_order), job))
            return
        if self.name == "mlfq":
            order = next(self.ready_order)
            if front:
                # Match insertion before the first waiting job at this level.
                same_level = [item[1] for item in self.ready_heap
                              if item[0] == job.level]
                if same_level:
                    first = min(same_level)
                    preceding = [item[1] for item in self.ready_heap
                                 if item[1] < first]
                    order = ((Fraction(max(preceding)) + Fraction(first)) / 2
                             if preceding else first - 1)
            heapq.heappush(self.ready_heap, (job.level, order, job))
            return
        if self.name == "pdb":
            queue = self.high_pending if job.high else self.low_ready
            if front:
                queue.insert(0, job)
            else:
                queue.append(job)
            return
        self.ready.append(job)

    @staticmethod
    def wait(job, now):
        return now - job.arrival - job.served

    def remaining_estimate(self, job):
        """Smooth residual service, once per step, using only observed work.

        Weight 1 replays raw estimates exactly. Otherwise negative observations
        are bounded at zero. Initial zero estimates remain a cold-start signal;
        the first positive observation seeds the smoother without dilution.
        Waiting does not consume service or update the estimate.
        """
        raw = float(self.traces[job.trace_id][job.step].get("est_rm", 0.0))
        if self.srtf_smoothing == 1:
            return raw
        if job.estimate_step != job.step:
            observed = max(0.0, raw)
            if job.smoothed_remaining is None:
                if observed > 0:
                    job.smoothed_remaining = observed
            else:
                consumed = max(0.0, job.served - job.estimate_served)
                residual = max(0.0, job.smoothed_remaining - consumed)
                weight = self.srtf_smoothing
                job.smoothed_remaining = (1 - weight) * residual + weight * observed
            job.estimate_step = job.step
            job.estimate_served = job.served
        return job.smoothed_remaining if job.smoothed_remaining is not None else 0.0

    def pick(self, now):
        if self.name == "pdb":
            if self.high_batch:
                return self.high_batch.pop(0)
            if self.paused is not None:
                job, self.paused = self.paused, None
                return job
            if self.low_ready:
                return self.low_ready.pop(0)
            return None
        elif self.name == "fcfs":
            return heapq.heappop(self.ready_heap)[1] if self.ready_heap else None
        elif self.name == "srtf":
            return heapq.heappop(self.ready_heap)[2] if self.ready_heap else None
        elif self.name == "mlfq":
            return heapq.heappop(self.ready_heap)[2] if self.ready_heap else None
        elif not self.ready:
            return None
        return self.ready.pop(0)

    def account_service(self, job, elapsed):
        if self.name == "pdb" and job.high and self.paused is not None:
            self.paused.pdb_high_service += elapsed

    def after_run(self, job, finished, quantum_expired=False):
        if finished:
            return
        if self.name == "mlfq":
            if quantum_expired:
                job.level = min(job.level + 1, len(self.quanta) - 1)
                job.quantum_used = 0.0
            self.add(job)
            return
        if self.name == "pdb":
            if job.high:
                self.high_batch.insert(0, job)
            else:
                self.low_ready.insert(0, job)
            return
        self.add(job)

    def boost(self):
        """Move all waiting MLFQ jobs to Q0, preserving their FIFO order."""
        for _, _, job in self.ready_heap:
            job.level = 0
            job.quantum_used = 0.0
        self.ready_heap = [(0, order, job) for _, order, job in self.ready_heap]
        heapq.heapify(self.ready_heap)

    def starvation_level(self, job, now):
        if job is None or job.pdb_start is None or now <= job.pdb_start:
            return 0.0
        return job.pdb_high_service / (now - job.pdb_start)

    def admit_pdb_batch(self, now, active):
        """Admit one bounded batch and return whether active low was preempted."""
        if self.name != "pdb" or not self.high_pending or self.high_batch:
            return False
        if active is not None and active.high:
            return False
        low = active if active is not None else self.paused
        if (low is not None and
                self.starvation_level(low, now) >= self.pdb_starvation_threshold):
            return False
        count = min(self.pdb_batch_size, len(self.high_pending))
        self.high_batch.extend(self.high_pending[:count])
        del self.high_pending[:count]
        if active is not None:
            self.paused = active
            return True
        return False

    def on_dispatch(self, job, now):
        if self.name == "pdb" and not job.high and job.pdb_start is None:
            job.pdb_start = now
            job.pdb_high_service = 0.0

    def on_pdb_completion(self, job, now):
        """End a batch early when its paused low request reaches Lmax."""
        if self.name != "pdb" or not job.high or self.paused is None:
            return
        if self.starvation_level(self.paused, now) >= self.pdb_starvation_threshold:
            self.high_pending[0:0] = self.high_batch
            self.high_batch.clear()


def simulate(traces, scheduler="srtf", load=0.5, warmup=1000.0,
             duration=10000.0, followup=10000.0, seed=47,
             include_profiling=False, alpha=0.1, mlfq_levels=4,
             mlfq_base_quantum=None, mlfq_boost_interval=None,
             pdb_batch_size=4, pdb_starvation_threshold=0.75,
             pdb_admission_interval=None, pdb_high_priority_fraction=0.2,
             pdb_high_trace_ids=None, arrival_rate=None, srtf_smoothing=0.2):
    """Measure arrivals in [warmup, warmup + duration), with ongoing follow-up.

    Arrivals continue until the entire measurement cohort finishes or the
    follow-up deadline is reached. Unfinished cohort members are censored and
    reported explicitly. No queue-draining phase changes their competition.
    """
    if scheduler not in {"fcfs", "srtf", "mlfq", "pdb"}:
        raise ValueError("Unknown scheduler")
    if arrival_rate is None:
        if not math.isfinite(load) or not 0 <= load <= 1:
            raise ValueError("Load must be between 0 and 1")
    elif not math.isfinite(arrival_rate) or arrival_rate < 0:
        raise ValueError("Arrival rate must be nonnegative and finite")
    if not all(math.isfinite(t) and t >= 0 for t in (warmup, duration, followup)) or duration == 0:
        raise ValueError("Duration must be positive; warmup/followup must be nonnegative and finite")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Aging alpha must be finite and nonnegative")
    if not isinstance(mlfq_levels, int) or mlfq_levels < 1:
        raise ValueError("MLFQ levels must be a positive integer")
    for value, label in ((mlfq_base_quantum, "MLFQ base quantum"),
                         (mlfq_boost_interval, "MLFQ boost interval")):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be positive and finite")
    if not isinstance(pdb_batch_size, int) or pdb_batch_size < 1:
        raise ValueError("PreemptDB batch size must be a positive integer")
    if (not math.isfinite(pdb_starvation_threshold) or
            not 0 <= pdb_starvation_threshold <= 1):
        raise ValueError("PreemptDB starvation threshold must be between 0 and 1")
    if (pdb_admission_interval is not None and
            (not math.isfinite(pdb_admission_interval) or pdb_admission_interval <= 0)):
        raise ValueError("PreemptDB admission interval must be positive and finite")
    if (not math.isfinite(pdb_high_priority_fraction) or
            not 0 <= pdb_high_priority_fraction <= 1):
        raise ValueError("PreemptDB high-priority fraction must be between 0 and 1")
    durations = service_times(traces, include_profiling)
    totals = list(map(sum, durations))
    if (pdb_high_trace_ids is not None and
            any(not isinstance(index, int) or index < 0 or index >= len(traces)
                for index in pdb_high_trace_ids)):
        raise ValueError("PreemptDB high-priority trace IDs are out of range")
    mean_service = statistics.mean(totals)
    control_mode = "arrival_rate" if arrival_rate is not None else "load"
    rate = arrival_rate if arrival_rate is not None else load / mean_service
    offered_load = rate * mean_service
    policy = Policy(scheduler, traces, durations, alpha, mlfq_levels,
                    mlfq_base_quantum, mlfq_boost_interval, pdb_batch_size,
                    pdb_starvation_threshold, pdb_admission_interval,
                    srtf_smoothing)
    # Separate streams keep workload selection independent of arrival sampling.
    arrivals_rng = random.Random(seed)
    traces_rng = random.Random(seed + 1)
    priorities_rng = random.Random(seed + 2)
    next_arrival = arrivals_rng.expovariate(rate) if rate else math.inf
    active = None
    next_finish = math.inf
    next_boost = policy.boost_interval if scheduler == "mlfq" else math.inf
    next_pdb_admission = math.inf
    now = 0.0
    end = warmup + duration
    deadline = end + followup
    cohort = {}
    completed = {}
    completed_priority = {}
    arrived = 0
    finished = 0
    measured_completions = 0
    measured_high_completions = 0
    measured_work = 0.0
    measured_high_arrivals = 0
    busy = 0.0
    backlog_at_end = None

    while True:
        boundary = end if backlog_at_end is None else deadline
        event_time = min(next_arrival, next_finish, next_boost,
                         next_pdb_admission, boundary)
        elapsed = event_time - now
        if active is not None:
            busy += max(0.0, min(event_time, end) - max(now, warmup))
            active.served += elapsed
            active.step_remaining = max(0.0, active.step_remaining - elapsed)
            if scheduler == "mlfq":
                active.quantum_used += elapsed
            policy.account_service(active, elapsed)
        now = event_time

        if next_finish == now:
            step_finished = active.step_remaining <= 1e-9
            quantum_expired = (scheduler == "mlfq" and
                               active.quantum_used >= policy.quanta[active.level] - 1e-9)
            if step_finished:
                active.step += 1
            done = active.step == len(durations[active.trace_id])
            if done:
                finished += 1
                if warmup <= now < end:
                    measured_completions += 1
                    measured_high_completions += int(active.high)
                if active.measured:
                    completed[active.qid] = now - active.arrival
                    completed_priority[active.qid] = active.high
                    del cohort[active.qid]
                policy.on_pdb_completion(active, now)
                if (scheduler == "pdb" and policy.high_pending and
                        math.isinf(next_pdb_admission)):
                    interval = policy.pdb_admission_interval
                    next_pdb_admission = (math.floor(now / interval) + 1) * interval
            else:
                if step_finished:
                    active.step_remaining = durations[active.trace_id][active.step]
                policy.after_run(active, False, quantum_expired)
            active = None
            next_finish = math.inf

        if next_boost == now:
            policy.boost()
            if active is not None:
                active.level = 0
                active.quantum_used = 0.0
                policy.add(active)
                active = None
                next_finish = math.inf
            next_boost += policy.boost_interval

        if next_pdb_admission == now:
            if policy.admit_pdb_batch(now, active):
                active = None
                next_finish = math.inf
            if policy.high_pending:
                next_pdb_admission += policy.pdb_admission_interval
            else:
                next_pdb_admission = math.inf

        if now == end and backlog_at_end is None:
            backlog_at_end = arrived - finished
        if backlog_at_end is not None and (not cohort or now >= deadline):
            break

        if next_arrival == now:
            trace_id = traces_rng.randrange(len(traces))
            measured = warmup <= now < end
            if scheduler == "pdb":
                high = (trace_id in pdb_high_trace_ids if pdb_high_trace_ids is not None
                        else priorities_rng.random() < pdb_high_priority_fraction)
            else:
                high = False
            job = Job(arrived, trace_id, now, measured, high,
                      step_remaining=durations[trace_id][0])
            policy.add(job)
            if scheduler == "pdb" and high and math.isinf(next_pdb_admission):
                interval = policy.pdb_admission_interval
                next_pdb_admission = (math.floor(now / interval) + 1) * interval
            arrived += 1
            if measured:
                cohort[job.qid] = job
                measured_work += totals[trace_id]
                measured_high_arrivals += int(high)
            next_arrival = now + arrivals_rng.expovariate(rate)
            # Standard MLFQ immediately preempts lower-priority work when a new
            # request arrives in Q0. The interrupted job keeps its remaining quantum.
            if scheduler == "mlfq" and active is not None and active.level > 0:
                policy.add(active, front=True)
                active = None
                next_finish = math.inf

        if active is None:
            active = policy.pick(now)
            if active is not None:
                policy.on_dispatch(active, now)
                run_for = active.step_remaining
                if scheduler == "mlfq":
                    run_for = min(run_for,
                                  policy.quanta[active.level] - active.quantum_used)
                next_finish = now + run_for
                if run_for > 0 and next_finish == now:
                    next_finish = math.nextafter(now, math.inf)

    jcts = [completed[qid] for qid in sorted(completed)]
    high_jcts = [completed[qid] for qid in sorted(completed)
                 if completed_priority[qid]]
    low_jcts = [completed[qid] for qid in sorted(completed)
                if not completed_priority[qid]]
    count = len(completed) + len(cohort)
    censored_high = sum(job.high for job in cohort.values())
    return {
        "scheduler": scheduler, "control_mode": control_mode,
        "load": offered_load, "load_percent": 100 * offered_load,
        "requested_load": load if control_mode == "load" else None,
        "requested_arrival_rate": rate if control_mode == "arrival_rate" else None,
        "lambda": rate, "mean_service_time": mean_service,
        "service_time": "exec+prf" if include_profiling else "exec",
        "seed": seed, "aging_alpha": alpha if scheduler == "srtf" else None,
        "srtf_smoothing": srtf_smoothing if scheduler == "srtf" else None,
        "mlfq_levels": len(policy.quanta) if scheduler == "mlfq" else None,
        "mlfq_base_quantum": policy.quanta[0] if scheduler == "mlfq" else None,
        "mlfq_quanta": policy.quanta if scheduler == "mlfq" else None,
        "mlfq_boost_interval": policy.boost_interval if scheduler == "mlfq" else None,
        "pdb_batch_size": policy.pdb_batch_size if scheduler == "pdb" else None,
        "pdb_starvation_threshold": (policy.pdb_starvation_threshold
                                      if scheduler == "pdb" else None),
        "pdb_admission_interval": (policy.pdb_admission_interval
                                    if scheduler == "pdb" else None),
        "pdb_priority_mode": ("trace_ids" if pdb_high_trace_ids is not None else "random_tags")
                             if scheduler == "pdb" else None,
        "pdb_high_priority_fraction": (pdb_high_priority_fraction
                                        if scheduler == "pdb" and pdb_high_trace_ids is None
                                        else None),
        "pdb_high_trace_ids": (sorted(pdb_high_trace_ids)
                               if scheduler == "pdb" and pdb_high_trace_ids is not None
                               else None),
        "warmup_seconds": warmup, "measurement_seconds": duration,
        "followup_limit_seconds": followup, "simulation_end_seconds": now,
        "steady_state_eligible": offered_load < 1,
        "measurement_arrivals": count, "completed_measurement_arrivals": len(jcts),
        "censored_count": len(cohort), "cohort_complete": not cohort,
        "measurement_high_arrivals": measured_high_arrivals,
        "measurement_low_arrivals": count - measured_high_arrivals,
        "censored_high_count": censored_high,
        "censored_low_count": len(cohort) - censored_high,
        "censored_latency_lower_bounds": [now - j.arrival for j in cohort.values()],
        "backlog_at_measurement_end": backlog_at_end,
        "backlog_at_stop": arrived - finished,
        "observed_utilization": busy / duration,
        "observed_offered_load": measured_work / duration,
        "throughput": measured_completions / duration,
        "high_throughput": (measured_high_completions / duration
                            if scheduler == "pdb" else None),
        "low_throughput": ((measured_completions - measured_high_completions) / duration
                           if scheduler == "pdb" else None),
        "jcts": jcts,
        "high_jcts": high_jcts if scheduler == "pdb" else None,
        "low_jcts": low_jcts if scheduler == "pdb" else None,
        # These statistics describe completed cohort members if censoring exists.
        "mean_jct": statistics.mean(jcts) if jcts else None,
        "p50_jct": percentile(jcts, 50), "p95_jct": percentile(jcts, 95),
        "p99_jct": percentile(jcts, 99),
        "high_p50_jct": percentile(high_jcts, 50) if scheduler == "pdb" else None,
        "high_p99_jct": percentile(high_jcts, 99) if scheduler == "pdb" else None,
        "low_p50_jct": percentile(low_jcts, 50) if scheduler == "pdb" else None,
        "low_p99_jct": percentile(low_jcts, 99) if scheduler == "pdb" else None,
    }


def main(scheduler):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Continuous-arrival {scheduler.upper()} load sweep (virtual seconds)")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--loads", type=float, nargs="+", metavar="PERCENT",
                          help="Normalized offered-load percentages from 0 to 100")
    controls.add_argument("--arrival-rates", type=float, nargs="+", metavar="QPS",
                          help="Direct Poisson arrival rates in queries/second; may overload the worker")
    parser.add_argument("--warmup", type=float, default=1000.0)
    parser.add_argument("--duration", type=float, default=10000.0)
    parser.add_argument("--followup", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--include-profiling", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1, help="SRTF aging weight; use 0 for pure SRTF")
    parser.add_argument("--srtf-smoothing", type=float, default=0.2,
                        help="New residual-estimate weight in (0, 1]; 1 replays raw SRTF")
    parser.add_argument("--mlfq-levels", type=int, default=4)
    parser.add_argument("--mlfq-base-quantum", type=float,
                        help="Q0 time quantum; defaults to the median positive trace-step time")
    parser.add_argument("--mlfq-boost-interval", type=float,
                        help="Global priority-boost period; defaults to 20x mean request service time")
    parser.add_argument("--pdb-batch-size", type=int, default=4)
    parser.add_argument("--pdb-starvation-threshold", type=float, default=0.75)
    parser.add_argument("--pdb-admission-interval", type=float,
                        help="High-priority admission period; defaults to median positive trace-step time")
    parser.add_argument("--pdb-high-priority-fraction", type=float, default=0.2,
                        help="Fraction tagged high priority when --pdb-high-traces is omitted")
    parser.add_argument("--pdb-high-traces", nargs="+",
                        help="Trace labels tagged high priority, e.g. t0 t1")
    parser.add_argument("--a", nargs="+", default=["0.2", "0.8"], help="Trace parameter groups")
    parser.add_argument("--b", nargs="+", default=["l", "m", "h"], help="Trace parameter groups")
    parser.add_argument("--trace-dir", type=Path, default=root / "traces" / "opt")
    parser.add_argument("--trace-suffix", default="",
                        help="Optional filename suffix before .json, such as _pd")
    parser.add_argument("--output-dir", type=Path, default=root / "sim" / "res")
    args = parser.parse_args()
    if not math.isfinite(args.srtf_smoothing) or not 0 < args.srtf_smoothing <= 1:
        parser.error("--srtf-smoothing must be in (0, 1]")
    if args.loads is None and args.arrival_rates is None:
        args.loads = list(range(0, 101, 10))
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.loads is not None and any(
            not math.isfinite(v) or not 0 <= v <= 100 for v in args.loads):
        parser.error("--loads must be percentages between 0 and 100")
    if args.arrival_rates is not None and any(
            not math.isfinite(v) or v < 0 for v in args.arrival_rates):
        parser.error("--arrival-rates must be nonnegative and finite")
    if (not all(math.isfinite(v) and v >= 0 for v in (args.warmup, args.duration, args.followup, args.alpha))
            or args.duration == 0):
        parser.error("Times and alpha must be finite and nonnegative; duration must be positive")
    if args.mlfq_levels < 1:
        parser.error("--mlfq-levels must be positive")
    for value, flag in ((args.mlfq_base_quantum, "--mlfq-base-quantum"),
                        (args.mlfq_boost_interval, "--mlfq-boost-interval")):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{flag} must be positive and finite")
    if args.pdb_batch_size < 1:
        parser.error("--pdb-batch-size must be positive")
    if (not math.isfinite(args.pdb_starvation_threshold) or
            not 0 <= args.pdb_starvation_threshold <= 1):
        parser.error("--pdb-starvation-threshold must be between 0 and 1")
    if (args.pdb_admission_interval is not None and
            (not math.isfinite(args.pdb_admission_interval) or
             args.pdb_admission_interval <= 0)):
        parser.error("--pdb-admission-interval must be positive and finite")
    if (not math.isfinite(args.pdb_high_priority_fraction) or
            not 0 <= args.pdb_high_priority_fraction <= 1):
        parser.error("--pdb-high-priority-fraction must be between 0 and 1")

    # Validate every requested group before starting the sweep or writing results.
    groups = []
    for a in args.a:
        for b in args.b:
            paths = sorted(args.trace_dir.glob(f"*_{a}_{b}{args.trace_suffix}.json"))
            if not paths:
                parser.error(f"No traces for a={a}, b={b} in {args.trace_dir}")
            try:
                traces = [json.loads(path.read_text()) for path in paths]
                service_times(traces, args.include_profiling)
            except (ValueError, TypeError, AttributeError, OSError) as exc:
                parser.error(f"Invalid traces for a={a}, b={b}: {exc}")
            high_trace_ids = None
            if scheduler == "pdb" and args.pdb_high_traces:
                labels = set(args.pdb_high_traces)
                high_trace_ids = {
                    index for index, path in enumerate(paths)
                    if path.name in labels or path.stem in labels
                    or path.name.split("_", 1)[0] in labels
                }
                matched = {
                    label for label in labels
                    if any(path.name == label or path.stem == label
                           or path.name.split("_", 1)[0] == label for path in paths)
                }
                if matched != labels:
                    parser.error(f"Unknown high-priority traces for a={a}, b={b}: "
                                 f"{sorted(labels - matched)}")
            groups.append((a, b, paths, traces, high_trace_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for a, b, paths, traces, high_trace_ids in groups:
        direct_rates = args.arrival_rates is not None
        output_kind = "rate" if direct_rates else "load"
        output = args.output_dir / f"{scheduler}_{a}_{b}_{output_kind}.jsonl"
        with output.open("a") as stream:
            controls_to_run = args.arrival_rates if direct_rates else args.loads
            for control_value in controls_to_run:
                for repeat in range(args.repeats):
                    result = simulate(
                        traces, scheduler,
                        0.0 if direct_rates else control_value / 100,
                        args.warmup,
                        args.duration, args.followup, args.seed + repeat,
                        args.include_profiling, args.alpha, args.mlfq_levels,
                        args.mlfq_base_quantum, args.mlfq_boost_interval,
                        args.pdb_batch_size, args.pdb_starvation_threshold,
                        args.pdb_admission_interval,
                        args.pdb_high_priority_fraction, high_trace_ids,
                        control_value if direct_rates else None,
                        srtf_smoothing=args.srtf_smoothing,
                    )
                    result.update(a=a, b=b, repeat=repeat, trace_files=[str(p.resolve()) for p in paths])
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    def metric_text(value):
                        return f"{value:.4f}s" if value is not None else "n/a"

                    control_text = (f"lambda={control_value:g}q/s "
                                    f"(offered_load={result['load_percent']:.2f}%)"
                                    if direct_rates else f"load={control_value:g}%")
                    print(f"{scheduler} {a}/{b} {control_text} "
                          f"mean={metric_text(result['mean_jct'])} "
                          f"p50={metric_text(result['p50_jct'])} "
                          f"p99={metric_text(result['p99_jct'])} "
                          f"arrivals={result['measurement_arrivals']} "
                          f"censored={result['censored_count']} -> {output}")
