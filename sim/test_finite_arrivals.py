import unittest

from sim.finite_arrivals import check_workload, make_workload, simulate_finite


class FiniteArrivalsTests(unittest.TestCase):
    def test_fcfs_drains_exactly_the_supplied_tasks(self):
        pools = [[[{'exec': 1.0}]] for _ in range(9)]
        workload = {'arrival_rate_qps': 1.0, 'tasks': [
            {'arrival': 0.0, 'plan': 0, 'repeat_draw': 0.0, 'high_priority': False},
            {'arrival': 0.5, 'plan': 0, 'repeat_draw': 0.0, 'high_priority': False},
            {'arrival': 0.5, 'plan': 0, 'repeat_draw': 0.0, 'high_priority': False},
        ]}
        result = simulate_finite(pools, 'fcfs', workload)
        self.assertEqual(result['tasks'], 3)
        self.assertEqual(result['jcts'], [1.0, 1.5, 2.5])
        self.assertEqual(result['completion_time'], 3.0)
        self.assertEqual(result['p50_jct'], 1.5)
        self.assertEqual(result['p90_jct'], 2.3)

    def test_same_workload_replays_across_policies(self):
        workload = make_workload(0.5, 200, 47, 'bio', '0.2_h')
        self.assertEqual(workload, make_workload(0.5, 200, 47, 'bio', '0.2_h'))
        check_workload(workload, 'bio', '0.2_h', 0.5, 200, 47)
        pools = [[[{'exec': 1.0}]] for _ in range(9)]
        for policy in ('fcfs', 'srtf', 'mlfq', 'pdb'):
            result = simulate_finite(pools, policy, workload)
            self.assertEqual(len(result['jcts']), 200)
            self.assertTrue(all(latency >= 1.0 - 1e-9 for latency in result['jcts']))


if __name__ == '__main__':
    unittest.main()
