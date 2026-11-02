"""Integration checks for the actual Bullet model and motor replay."""

import unittest

from panda_sorting import BINS, HOME, PICKS, BulletCell, plan, replay, train


class PandaSortingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = BulletCell()
        cls.cell.build_graph()

    @classmethod
    def tearDownClass(cls):
        cls.cell.close()

    def test_all_orders_reachable_and_carrying_requires_clearance(self):
        self.assertEqual(len(PICKS), 4)
        self.assertEqual(len(BINS), 4)
        for pick in PICKS:
            self.assertIsNotNone(self.cell.shortest(HOME, pick, 0))
            for bin_node in BINS:
                self.assertIsNotNone(self.cell.shortest(pick, bin_node, 1))
        empty = sum(map(len, self.cell.edges[0].values()))
        carrying = sum(map(len, self.cell.edges[1].values()))
        self.assertLess(carrying, empty)

    def test_trained_policy_executes_in_physics(self):
        agent, _ = train(self.cell, 2000, 42)
        for pickup in range(len(PICKS)):
            for bin_id in range(len(BINS)):
                task, path = plan(self.cell, agent, pickup, bin_id,
                                  42 + pickup * len(BINS) + bin_id)
                self.assertTrue(task.success)
                result = replay(self.cell, path, pickup, bin_id)
                self.assertTrue(result["physical_success"], result)
                self.assertEqual(result["robot_fixture_contact_events"], 0)


if __name__ == "__main__":
    unittest.main()
