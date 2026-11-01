"""Focused safety and replay checks for the teaching simulator."""

import unittest

from arm_sorting import ACTIONS, ArmEnv, BINS, HOME, PICKS, SAFE, motion_safe, replay, train


class ArmSortingTest(unittest.TestCase):
    def test_invalid_action_is_interlocked(self):
        env = ArmEnv(0, 0)
        env.q = (0, 0)
        before = env.q
        env.step(0)  # joint 1 would leave its allowed range
        self.assertEqual(env.q, before)
        self.assertEqual(env.interlocks, 1)

    def test_replays_follow_safe_continuous_commands(self):
        table, _ = train(4000, 42)
        for p in range(len(PICKS)):
            for b in range(len(BINS)):
                env, frames = replay(table, p, b, 42 + p * 2 + b)
                self.assertTrue(env.success, (p, b))
                for left, right in zip(frames, frames[1:]):
                    a, z = tuple(left["q"]), tuple(right["q"])
                    self.assertIn(z, SAFE)
                    if a != z:
                        self.assertTrue(motion_safe(a, z))


if __name__ == "__main__":
    unittest.main()
