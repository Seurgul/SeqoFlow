import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .load import Job, Policy, service_times, simulate
except ImportError:
    from load import Job, Policy, service_times, simulate


POLICIES = ("fcfs", "srtf", "mlfq", "pdb")
TRACES = [[{"exec": 0.3, "prf": 0.2, "est_rm": 1.0},
           {"exec": 0.7, "prf": 0.3, "est_rm": 0.7}]]


class ContinuousLoadTests(unittest.TestCase):
    def test_residual_smoothing_consumes_service_once_not_waiting(self):
        traces = [[{"exec": 10, "est_rm": 100},
                   {"exec": 2, "est_rm": 20},
                   {"exec": 3, "est_rm": -5}]]
        policy = Policy("srtf", traces, service_times(traces), alpha=0,
                        srtf_smoothing=0.2)
        job = Job(0, 0, 0, True, False)
        self.assertEqual(policy.remaining_estimate(job), 100)
        job.step, job.served = 1, 10
        self.assertAlmostEqual(policy.remaining_estimate(job), 76)
        # Repeated scheduling while this query waits must not smooth again.
        policy.add(job)
        self.assertIs(policy.pick(1000), job)
        self.assertAlmostEqual(policy.remaining_estimate(job), 76)
        job.step, job.served = 2, 12
        self.assertAlmostEqual(policy.remaining_estimate(job), 59.2)

    def test_smoothing_bootstrap_is_per_job_and_raw_mode_is_exact(self):
        traces = [[{"exec": 1, "est_rm": 0},
                   {"exec": 1, "est_rm": 100},
                   {"exec": 1, "est_rm": -2}]]
        policy = Policy("srtf", traces, service_times(traces), alpha=0)
        job = Job(0, 0, 0, True, False)
        other = Job(1, 0, 0, True, False)
        self.assertEqual(policy.remaining_estimate(job), 0)
        job.step, job.served = 1, 1
        self.assertEqual(policy.remaining_estimate(job), 100)
        self.assertEqual(policy.remaining_estimate(other), 0)
        raw = Policy("srtf", traces, service_times(traces), alpha=0,
                     srtf_smoothing=1)
        job.step = 2
        self.assertEqual(raw.remaining_estimate(job), -2)

    def test_smoothing_only_changes_srtf_and_is_reported(self):
        for name in ("fcfs", "mlfq", "pdb"):
            settings = dict(scheduler=name, warmup=10, duration=50, followup=100)
            self.assertEqual(simulate(TRACES, srtf_smoothing=0.2, **settings),
                             simulate(TRACES, srtf_smoothing=1, **settings))
        result = simulate(TRACES, warmup=10, duration=50, srtf_smoothing=0.4)
        self.assertEqual(result["srtf_smoothing"], 0.4)
        for weight in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                simulate(TRACES, srtf_smoothing=weight)

    def test_zero_load_has_no_latency_samples(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = simulate(TRACES, policy, load=0, warmup=10, duration=20)
                self.assertEqual(result["measurement_arrivals"], 0)
                self.assertEqual(result["jcts"], [])
                self.assertIsNone(result["mean_jct"])
                self.assertEqual(result["observed_utilization"], 0)
                self.assertEqual(result["simulation_end_seconds"], 30)

    def test_rate_uses_executed_service_definition(self):
        for profiling, expected in ((False, 1), (True, 1.5)):
            result = simulate(TRACES, load=0.75, warmup=10, duration=100,
                              include_profiling=profiling)
            self.assertEqual(result["mean_service_time"], expected)
            self.assertAlmostEqual(result["lambda"] * expected, 0.75)
            self.assertTrue(all(jct >= expected - 1e-10 for jct in result["jcts"]))

    def test_fcfs_matches_md1_mean_latency(self):
        # For unit deterministic service, E[T] = 1 + rho / (2 * (1-rho)).
        result = simulate(TRACES, "fcfs", load=0.5, warmup=2000,
                          duration=100000, followup=1000, seed=123)
        self.assertAlmostEqual(result["mean_jct"], 1.5, delta=0.04)
        self.assertAlmostEqual(result["observed_utilization"], 0.5, delta=0.01)
        self.assertEqual(result["censored_count"], 0)

    def test_measurement_is_timed_and_reproducible_across_policies(self):
        traces = TRACES + [[{"exec": 5, "est_rm": 5}]]
        results = [simulate(traces, policy, load=0.8, warmup=50,
                            duration=500, followup=5000) for policy in POLICIES]
        for policy, result in zip(POLICIES, results):
            self.assertEqual(result, simulate(traces, policy, load=0.8,
                                             warmup=50, duration=500, followup=5000))
            self.assertEqual(result["measurement_arrivals"], results[0]["measurement_arrivals"])
            self.assertEqual(result["observed_offered_load"], results[0]["observed_offered_load"])
            self.assertEqual(result["censored_count"], 0)
            self.assertEqual(len(result["jcts"]), result["measurement_arrivals"])
            self.assertLessEqual(result["observed_utilization"], 1)
        longer = simulate(traces, duration=1000, warmup=50, load=0.8)
        self.assertGreater(longer["measurement_arrivals"], results[0]["measurement_arrivals"])

    def test_saturation_is_bounded_and_reports_censoring(self):
        traces = [[{"exec": 100, "est_rm": 100}]]
        for policy in POLICIES:
            result = simulate(traces, policy, load=1, warmup=0,
                              duration=100, followup=0)
            self.assertEqual(result["simulation_end_seconds"], 100)
            self.assertGreater(result["measurement_arrivals"], 0)
            self.assertEqual(result["censored_count"], result["measurement_arrivals"])
            self.assertEqual(result["backlog_at_measurement_end"], result["censored_count"])
            self.assertEqual(len(result["censored_latency_lower_bounds"]), result["censored_count"])
            self.assertFalse(result["steady_state_eligible"])
            self.assertIsNone(result["mean_jct"])

    def test_followup_keeps_arrivals_running(self):
        traces = [[{"exec": 100, "est_rm": 100}]]
        result = simulate(traces, "fcfs", load=1, warmup=0,
                          duration=100, followup=1000)
        self.assertEqual(result["censored_count"], 0)
        self.assertGreater(result["simulation_end_seconds"], 100)
        # No request completes inside the measurement window, but requests that
        # arrived inside it complete during follow-up and belong to the latency
        # sample. This distinguishes arrival-cohort from completion-window sampling.
        self.assertEqual(result["throughput"], 0)
        self.assertGreater(result["measurement_arrivals"], 0)
        self.assertEqual(len(result["jcts"]), result["measurement_arrivals"])
        # Later arrivals remain when the original arrival cohort has finished.
        self.assertGreater(result["backlog_at_stop"], 0)

    def test_invalid_inputs_fail_early(self):
        for kwargs in ({"load": -0.1}, {"load": 1.1}, {"load": float("nan")},
                       {"duration": 0}, {"warmup": -1}, {"followup": float("inf")},
                       {"alpha": -1}, {"scheduler": "unknown"},
                       {"mlfq_levels": 0}, {"mlfq_base_quantum": 0},
                       {"mlfq_boost_interval": -1}, {"pdb_batch_size": 0},
                       {"pdb_starvation_threshold": 1.1},
                       {"pdb_admission_interval": 0},
                       {"pdb_high_priority_fraction": -0.1},
                       {"pdb_high_trace_ids": {99}},
                       {"arrival_rate": -0.1},
                       {"arrival_rate": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(TRACES, **kwargs)
        for traces in ([], [[]], [[{"exec": -1}]], [[{"exec": 0}]],
                       [[{"exec": float("nan")}]]):
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                service_times(traces)

    def test_zero_duration_steps_complete(self):
        traces = [[{"exec": 0}, {"exec": 1}], [{"exec": 0}]]
        for policy in POLICIES:
            result = simulate(traces, policy, warmup=10, duration=100, followup=1000)
            self.assertEqual(result["censored_count"], 0)

    def test_tiny_step_at_large_virtual_time_advances(self):
        traces = [[{"exec": 20_000_000}, {"exec": 1.5e-9}]]
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = simulate(traces, policy, arrival_rate=1e-8,
                                  warmup=0, duration=40_000_000,
                                  followup=0, seed=1)
                self.assertEqual(result["measurement_arrivals"], 1)
                self.assertEqual(result["completed_measurement_arrivals"], 1)
                self.assertEqual(result["censored_count"], 0)

    def test_srtf_distinguishes_positive_estimates(self):
        traces = [[{"exec": 10, "est_rm": 10}], [{"exec": 1, "est_rm": 1}]]
        policy = Policy("srtf", traces, service_times(traces), alpha=0)
        policy.add(Job(0, 0, 0, True, False))
        policy.add(Job(1, 1, 0, True, True))
        self.assertEqual(policy.pick(0).qid, 1)

    def test_mlfq_uses_fifo_geometric_quanta_demotion_and_boost(self):
        traces = [[{"exec": 10, "est_rm": 10}]]
        policy = Policy("mlfq", traces, service_times(traces), alpha=0.1,
                        mlfq_levels=3, mlfq_base_quantum=1,
                        mlfq_boost_interval=20)
        self.assertEqual(policy.quanta, [1, 2, 4])
        first = Job(0, 0, 0, True, False, step_remaining=10)
        second = Job(1, 0, 0, True, False, step_remaining=10)
        policy.add(first)
        policy.add(second)
        self.assertIs(policy.pick(0), first)
        first.quantum_used = 1
        policy.after_run(first, finished=False, quantum_expired=True)
        self.assertEqual(first.level, 1)
        self.assertEqual(first.quantum_used, 0)
        self.assertIs(policy.pick(1), second)
        policy.boost()
        self.assertEqual(first.level, 0)
        self.assertEqual(first.quantum_used, 0)

    def test_mlfq_configuration_is_reported(self):
        result = simulate(TRACES, "mlfq", load=0.5, warmup=20,
                          duration=200, followup=1000, mlfq_levels=3,
                          mlfq_base_quantum=0.25,
                          mlfq_boost_interval=10)
        self.assertEqual(result["mlfq_levels"], 3)
        self.assertEqual(result["mlfq_quanta"], [0.25, 0.5, 1.0])
        self.assertEqual(result["mlfq_boost_interval"], 10)
        self.assertEqual(result["censored_count"], 0)

    def test_preemptdb_bounds_batches_and_resumes_paused_low(self):
        traces = [[{"exec": 10, "est_rm": 10}]]
        policy = Policy("pdb", traces, service_times(traces), alpha=0.1,
                        pdb_batch_size=2, pdb_starvation_threshold=0.75,
                        pdb_admission_interval=1)
        low = Job(0, 0, 0, True, False, step_remaining=10)
        policy.on_dispatch(low, 0)
        highs = [Job(qid, 0, 0, True, True, step_remaining=1)
                 for qid in (1, 2, 3)]
        for job in highs:
            policy.add(job)
        self.assertTrue(policy.admit_pdb_batch(now=1, active=low))
        self.assertIs(policy.paused, low)
        self.assertEqual([job.qid for job in policy.high_batch], [1, 2])
        self.assertEqual([job.qid for job in policy.high_pending], [3])
        self.assertEqual(policy.pick(1).qid, 1)
        self.assertEqual(policy.pick(1).qid, 2)
        self.assertIs(policy.pick(1), low)

    def test_preemptdb_starvation_threshold_ends_batch_early(self):
        traces = [[{"exec": 10, "est_rm": 10}]]
        policy = Policy("pdb", traces, service_times(traces), alpha=0.1,
                        pdb_batch_size=3, pdb_starvation_threshold=0.75,
                        pdb_admission_interval=1)
        low = Job(0, 0, 0, True, False, step_remaining=10)
        policy.on_dispatch(low, 0)
        for qid in (1, 2, 3):
            policy.add(Job(qid, 0, 0, True, True, step_remaining=1))
        policy.admit_pdb_batch(now=1, active=low)
        first_high = policy.pick(1)
        policy.account_service(first_high, 1.6)
        policy.on_pdb_completion(first_high, now=2)
        self.assertEqual(policy.high_batch, [])
        self.assertEqual([job.qid for job in policy.high_pending], [2, 3])
        self.assertIs(policy.pick(2), low)

    def test_preemptdb_reports_external_priority_classes(self):
        result = simulate(TRACES, "pdb", load=0.5, warmup=20,
                          duration=200, followup=1000,
                          pdb_high_priority_fraction=1,
                          pdb_admission_interval=0.1)
        self.assertGreater(result["measurement_arrivals"], 0)
        self.assertEqual(result["measurement_high_arrivals"],
                         result["measurement_arrivals"])
        self.assertEqual(result["measurement_low_arrivals"], 0)
        self.assertEqual(result["high_jcts"], result["jcts"])
        self.assertEqual(result["low_jcts"], [])
        self.assertEqual(result["pdb_priority_mode"], "random_tags")

    def test_direct_arrival_rate_can_exceed_capacity(self):
        traces = [[{"exec": 1, "est_rm": 1}]]
        result = simulate(traces, "fcfs", warmup=10, duration=100,
                          followup=0, arrival_rate=1.1)
        self.assertEqual(result["control_mode"], "arrival_rate")
        self.assertEqual(result["requested_arrival_rate"], 1.1)
        self.assertAlmostEqual(result["load"], 1.1)
        self.assertAlmostEqual(result["load_percent"], 110)
        self.assertFalse(result["steady_state_eligible"])
        self.assertGreater(result["backlog_at_measurement_end"], 0)

    def test_all_cli_entrypoints_from_another_working_directory(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for suffix in ("", "_pd"):
                (tmp / f"t0_0.8_m{suffix}.json").write_text(json.dumps(TRACES[0]))
            for script, policy in (("fcfs", "fcfs"), ("srtf", "srtf"),
                                   ("mlfq", "mlfq"), ("preempt", "pdb")):
                proc = subprocess.run(
                    [sys.executable, str(root / f"{script}.py"), "--trace-dir", str(tmp),
                     "--output-dir", str(tmp / "out"), "--a", "0.8", "--b", "m",
                     "--loads", "0", "50", "100", "--warmup", "10", "--duration", "100",
                     "--followup", "200", "--repeats", "2"],
                    cwd=tmp, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                rows = [json.loads(line) for line in (tmp / "out" / f"{policy}_0.8_m_load.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 6)
                self.assertEqual([row["load_percent"] for row in rows], [0, 0, 50, 50, 100, 100])


if __name__ == "__main__":
    unittest.main()
