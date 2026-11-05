"""Finite Poisson-arrival experiments with reusable workloads.

Examples (from the repository root)::

    python -m sim.finite_arrivals --dataset bio --a 0.2 --b h \
        --arrival-rates 0.001 0.003 0.005 --tasks 200
    python -m sim.finite_arrivals --dataset legal --arrival-rates 0.001 0.002

One workload JSON is saved per rate and seed. Repeating the command reuses it;
all methods receive the same arrival times, plan IDs, and repeat draws.
Existing raw results are skipped; only missing method/rate/seed combinations run.
The latency summary retains results from earlier invocations.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

from sim.baseline_data import BIO_SETTINGS, ROOT, load_plan_latencies
from sim.load import Job, Policy, percentile, service_times


METHODS = {
    'seqoflow:fcfs': ('seqoflow', 'fcfs'),
    'seqoflow:srtf': ('seqoflow', 'srtf'),
    'seqoflow:mlfq': ('seqoflow', 'mlfq'),
    'seqoflow:pdb': ('seqoflow', 'pdb'),
    'pd:srtf': ('pd', 'srtf'),
    'pu:srtf': ('pu', 'srtf'),
    'lotus:fcfs': ('lotus', 'fcfs'),
    'eddies:fcfs': ('eddies', 'fcfs'),
}
SUFFIX = {'seqoflow': '', 'pd': '_pd', 'pu': '_pu'}
FIELDS = ('dataset', 'setting', 'method', 'arrival_rate_qps', 'seed', 'tasks',
          'mean_service_seconds', 'offered_load_percent', 'p50_jct', 'p99_jct')


def make_workload(rate, count, seed, dataset, setting):
    """Draw plan IDs and repeats independently of the arrival process."""
    if not math.isfinite(rate) or rate <= 0 or count < 1:
        raise ValueError('arrival rate and task count must be positive')
    arrivals = random.Random(seed)
    plans = random.Random(seed + 1)
    repeats = random.Random(seed + 2)
    priorities = random.Random(seed + 3)
    now = 0.0
    tasks = []
    for _ in range(count):
        now += arrivals.expovariate(rate)
        tasks.append({'arrival': now, 'plan': plans.randrange(9),
                      'repeat_draw': repeats.random(),
                      'high_priority': priorities.random() < 0.2})
    return {'dataset': dataset, 'setting': setting, 'arrival_rate_qps': rate,
            'seed': seed, 'tasks': tasks}


def check_workload(workload, dataset, setting, rate, count, seed):
    if (workload.get('dataset') != dataset or workload.get('setting') != setting
            or workload.get('arrival_rate_qps') != rate
            or workload.get('seed') != seed or len(workload.get('tasks', [])) != count):
        raise ValueError('saved workload does not match requested dataset, setting, rate, seed, or task count')
    previous = -math.inf
    for task in workload['tasks']:
        arrival, plan, draw = task['arrival'], task['plan'], task['repeat_draw']
        if (not isinstance(arrival, (int, float)) or not math.isfinite(arrival)
                or arrival < previous or arrival < 0
                or not isinstance(plan, int) or not 0 <= plan < 9
                or not isinstance(draw, (int, float)) or not 0 <= draw < 1
                or not isinstance(task.get('high_priority'), bool)):
            raise ValueError('saved workload has invalid task arrivals or selections')
        previous = arrival


def load_pools(dataset, setting, method):
    """Return nine plan pools; baseline runs are whole-job service times."""
    workload, _ = METHODS[method]
    if workload in ('lotus', 'eddies'):
        samples = load_plan_latencies(workload, setting, dataset)
        return [[[{'exec': latency}] for latency in samples[plan]]
                for plan in range(9)]

    directory = ROOT / 'sim/traces' / ('biodex' if dataset == 'bio' else dataset)
    pools = []
    for plan in range(9):
        paths = (sorted(directory.glob(f't{plan}_run*_{setting}{SUFFIX[workload]}.json'))
                 if dataset == 'bio' else [directory / f't{plan}{SUFFIX[workload]}.json'])
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError(f'missing {method} trace for t{plan} in {directory}')
        pools.append([json.loads(path.read_text()) for path in paths])
    return pools


def simulate_finite(pools, scheduler, workload):
    """Run exactly the supplied arrivals, then drain the single worker."""
    if scheduler not in ('fcfs', 'srtf', 'mlfq', 'pdb') or len(pools) != 9 or any(not p for p in pools):
        raise ValueError('invalid scheduler or incomplete trace pools')
    traces = [trace for pool in pools for trace in pool]
    durations = service_times(traces, include_profiling=True)
    offsets = []
    index = 0
    for pool in pools:
        offsets.append(index)
        index += len(pool)
    policy = Policy(scheduler, traces, durations, alpha=0.1, srtf_smoothing=0.2)
    tasks = workload['tasks']
    arrivals = 0
    completed = {}
    active = None
    next_finish = math.inf
    next_boost = policy.boost_interval if scheduler == 'mlfq' else math.inf
    next_pdb_admission = math.inf
    now = 0.0

    while len(completed) < len(tasks):
        next_arrival = tasks[arrivals]['arrival'] if arrivals < len(tasks) else math.inf
        event_time = min(next_arrival, next_finish, next_boost, next_pdb_admission)
        if not math.isfinite(event_time):
            raise RuntimeError('unfinished tasks with no future simulator event')
        elapsed = event_time - now
        if active is not None:
            active.served += elapsed
            active.step_remaining = max(0.0, active.step_remaining - elapsed)
            if scheduler == 'mlfq':
                active.quantum_used += elapsed
            policy.account_service(active, elapsed)
        now = event_time

        if next_finish == now:
            step_finished = active.step_remaining <= 1e-9
            quantum_expired = (scheduler == 'mlfq' and
                               active.quantum_used >= policy.quanta[active.level] - 1e-9)
            if step_finished:
                active.step += 1
            if active.step == len(durations[active.trace_id]):
                completed[active.qid] = now - active.arrival
                policy.on_pdb_completion(active, now)
                if scheduler == 'pdb' and policy.high_pending and math.isinf(next_pdb_admission):
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
            next_pdb_admission = (next_pdb_admission + policy.pdb_admission_interval
                                  if policy.high_pending else math.inf)

        if next_arrival == now:
            task = tasks[arrivals]
            plan = task['plan']
            trace_id = offsets[plan] + int(task['repeat_draw'] * len(pools[plan]))
            job = Job(arrivals, trace_id, now, True,
                      task['high_priority'] if scheduler == 'pdb' else False,
                      step_remaining=durations[trace_id][0])
            policy.add(job)
            if scheduler == 'pdb' and job.high and math.isinf(next_pdb_admission):
                interval = policy.pdb_admission_interval
                next_pdb_admission = (math.floor(now / interval) + 1) * interval
            arrivals += 1
            if scheduler == 'mlfq' and active is not None and active.level > 0:
                policy.add(active, front=True)
                active = None
                next_finish = math.inf

        if active is None:
            active = policy.pick(now)
            if active is not None:
                policy.on_dispatch(active, now)
                run_for = active.step_remaining
                if scheduler == 'mlfq':
                    run_for = min(run_for, policy.quanta[active.level] - active.quantum_used)
                next_finish = now + run_for

    jcts = [completed[i] for i in range(len(tasks))]
    average = mean(map(sum, durations))
    return {'scheduler': scheduler, 'tasks': len(tasks),
            'arrival_rate_qps': workload['arrival_rate_qps'],
            'mean_service_seconds': average,
            'offered_load_percent': 100 * workload['arrival_rate_qps'] * average,
            'completion_time': now, 'jcts': jcts,
            'p50_jct': percentile(jcts, 50), 'p99_jct': percentile(jcts, 99)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('bio', 'legal', 'patent'), default='bio')
    parser.add_argument('--a', default='0.2', help='BioDEX selectivity setting')
    parser.add_argument('--b', default='h', help='BioDEX profile setting')
    parser.add_argument('--arrival-rates', type=float, nargs='+', required=True,
                        help='Poisson arrival rates in queries per second')
    parser.add_argument('--tasks', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=(47,))
    parser.add_argument('--methods', nargs='+', choices=tuple(METHODS), default=tuple(METHODS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workload-dir', type=Path,
                        help='Read or save reusable workload JSON files here')
    parser.add_argument('--generate-only', action='store_true',
                        help='Save workload JSON files without running methods')
    args = parser.parse_args()
    setting = f'{args.a}_{args.b}' if args.dataset == 'bio' else args.dataset
    if args.dataset == 'bio' and setting not in BIO_SETTINGS:
        parser.error(f'BioDEX setting must be one of {BIO_SETTINGS}')
    if (args.tasks < 1 or any(not math.isfinite(r) or r <= 0 for r in args.arrival_rates)
            or len(set(args.arrival_rates)) != len(args.arrival_rates)
            or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.methods)) != len(args.methods)):
        parser.error('tasks and rates must be positive; rates, seeds, and methods must be unique')
    output = args.output_dir or ROOT / 'sim/results/finite_arrivals' / args.dataset / setting
    workload_dir = args.workload_dir or output / 'workloads'
    raw_dir = output / 'raw'
    pools = {}
    workload_dir.mkdir(parents=True, exist_ok=True)
    if not args.generate_only:
        raw_dir.mkdir(parents=True, exist_ok=True)
    summary = output / 'latency_summary.csv'

    def row_key(row):
        return (row['dataset'], row['setting'], row['method'],
                float(row['arrival_rate_qps']), int(row['seed']), int(row['tasks']))

    rows = {}
    if not args.generate_only and summary.exists():
        with summary.open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows[row_key(row)] = {field: row[field] for field in FIELDS}
    for rate in sorted(args.arrival_rates):
        for seed in args.seeds:
            pending = []
            if not args.generate_only:
                for method in args.methods:
                    result_path = raw_dir / f'{method.replace(":", "_")}_rate_{rate:g}_seed_{seed}.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        rows[row_key(result)] = {field: result[field] for field in FIELDS}
                        print(f'Skipping {method} rate={rate:g} seed={seed}: result exists', flush=True)
                    else:
                        pending.append((method, result_path))
                if not pending:
                    continue
            workload_path = workload_dir / f'rate_{rate:g}_seed_{seed}_n_{args.tasks}.json'
            if workload_path.exists():
                workload = json.loads(workload_path.read_text())
            else:
                workload = make_workload(rate, args.tasks, seed, args.dataset, setting)
                workload_path.write_text(json.dumps(workload) + '\n')
            check_workload(workload, args.dataset, setting, rate, args.tasks, seed)
            if args.generate_only:
                print(f'Saved {workload_path}')
                continue
            for method, result_path in pending:
                if method not in pools:
                    try:
                        pools[method] = load_pools(args.dataset, setting, method)
                        service_times([trace for pool in pools[method] for trace in pool], True)
                    except (ValueError, OSError, json.JSONDecodeError) as exc:
                        parser.error(str(exc))
                result = simulate_finite(pools[method], METHODS[method][1], workload)
                result.update(dataset=args.dataset, setting=setting, method=method,
                              seed=seed, workload_file=str(workload_path))
                result_path.write_text(json.dumps(result) + '\n')
                rows[row_key(result)] = {field: result[field] for field in FIELDS}
                print(f'{method} rate={rate:g} seed={seed} '
                      f'p50={result["p50_jct"]:.3f}s p99={result["p99_jct"]:.3f}s', flush=True)
    if args.generate_only:
        return
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row for _, row in sorted(rows.items()))
    print(f'Saved {len(rows)} results to {summary}')


if __name__ == '__main__':
    main()
