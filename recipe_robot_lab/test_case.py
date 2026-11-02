"""Contract and end-to-end checks for the simulated lab case."""
from __future__ import annotations

import unittest

import numpy as np

from calibration import fit_model, generate_synthetic_rows
from compiler import DEFAULT_TEXT, compile_actions, compile_text
from dose_rl import train
from driver import DeviceFault, SimLabDriver
from evaluation import evaluate_dose_policies
from motion_rl import BLOCKS, STATION_NODES, plan, train as train_motion
from run_case import experiment, run_attempt
from twin import BufferModel


class RecipeLabTests(unittest.TestCase):
    def test_compiler_requires_explicit_supported_reagents_and_units(self):
        self.assertEqual(compile_text(DEFAULT_TEXT).final_volume_ml, 20)
        with self.assertRaisesRegex(ValueError, "只支持"):
            compile_text(DEFAULT_TEXT.replace("柠檬酸钠", "盐酸"))
        with self.assertRaisesRegex(ValueError, "浓度"):
            compile_text(DEFAULT_TEXT.replace("0.1 mol/L柠檬酸钠", "柠檬酸钠"))
        for instruction in ("另加入5 mL盐酸。", "将样品加热到80℃。"):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, "未识别"):
                compile_text(DEFAULT_TEXT + instruction)

    def test_protocol_enforces_capacity_and_station_sequence(self):
        protocol = compile_text(DEFAULT_TEXT)
        actions = compile_actions(protocol, 6, 9, 20)
        self.assertEqual([x["station"] for x in actions if x["op"] == "move"],
                         ["doser", "mixer", "ph_meter", "doser", "mixer", "ph_meter", "rack"])
        with self.assertRaisesRegex(ValueError, "超过"):
            compile_actions(protocol, 10, 10, 20)

    def test_simulated_cycle_closes_volume_and_records_arm_feedback(self):
        protocol = compile_text(DEFAULT_TEXT)
        policy, _ = train(protocol, episodes=80, seed=17)
        motion_policy, motion_metrics = train_motion(episodes=100, seed=17)
        self.assertGreaterEqual(motion_metrics["station_pair_policy_success"], 0.9)
        rng = np.random.default_rng(123)
        driver = SimLabDriver(protocol, 6.0, 9.0, 20, rng, bias=0.03,
                              motion_policy=motion_policy)
        try:
            result = experiment(protocol, (6.0, 9.0, 20), policy, driver, index=1)
        finally:
            driver.close()
        self.assertEqual(result["mode"], "simulation")
        self.assertAlmostEqual(result["final_volume_ml"], 20.0, places=5)
        self.assertLess(result["max_arm_position_error_m"], 0.025)
        self.assertTrue(result["readings"])
        self.assertTrue(all(x["stable"] and len(x["raw_ph"]) >= 3 for x in result["readings"]))
        self.assertIn("ph_after_10min", [x["kind"] for x in result["readings"]])
        self.assertGreater(result["elapsed_s"], 0)
        self.assertGreater(result["motion_policy_moves"], 0)
        self.assertEqual(result["unsafe_station_contacts"], 0)
        self.assertTrue(any(x["event"] == "vial_at_station" for x in result["arm_trace"]))
        self.assertTrue(any(x["event"] == "release_vial" for x in result["arm_trace"]))
        self.assertTrue(any(x["event"] == "rl_decision" and len(x["q_values"]) == 7
                            for x in result["events"]))

    def test_sample_changes_only_after_dispense_and_requires_mix(self):
        protocol = compile_text(DEFAULT_TEXT)
        driver = SimLabDriver(protocol, 6.0, 9.0, 20, np.random.default_rng(9), bias=0.03)
        try:
            self.assertEqual(driver.state()["total_volume_ml"], 0)
            driver.grasp()
            driver.move("doser")
            driver.dispense("citric_acid", 6.0)
            self.assertGreater(driver.state()["total_volume_ml"], 0)
            driver.move("ph_meter")
            with self.assertRaisesRegex(RuntimeError, "尚未混合"):
                driver.measure_ph("ph")
        finally:
            driver.close()

    def test_injected_instrument_fault_triggers_safe_stop(self):
        protocol = compile_text(DEFAULT_TEXT)
        policy, _ = train(protocol, episodes=80, seed=17)
        driver = SimLabDriver(protocol, 6.0, 9.0, 20, np.random.default_rng(9),
                              bias=0.03, fault_at=("measure", 1))
        try:
            with self.assertRaises(DeviceFault) as caught:
                experiment(protocol, (6.0, 9.0, 20), policy, driver, index=1)
            driver.abort(str(caught.exception))
            self.assertEqual(driver.trace[-1]["event"], "safe_stop")
            self.assertTrue(driver.trace[-1]["vial_held"])
        finally:
            driver.close()

    def test_unexpected_measurement_error_is_stopped_and_recorded(self):
        protocol = compile_text(DEFAULT_TEXT)
        policy, _ = train(protocol, episodes=80, seed=17)
        driver = SimLabDriver(protocol, 6.0, 9.0, 20, np.random.default_rng(9),
                              bias=0.03)
        try:
            def fail_measure(_kind):
                raise RuntimeError("模拟 pH 电极不稳定")
            driver.measure_ph = fail_measure
            row = run_attempt(protocol, (6.0, 9.0, 20), policy, driver, index=1)
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["fault"]["type"], "RuntimeError")
            self.assertTrue(row["safe_stop_confirmed"])
            self.assertEqual(row["arm_trace"][-1]["event"], "safe_stop")
            self.assertGreater(row["state_at_abort"]["total_volume_ml"], 0)
        finally:
            driver.close()

    def test_safe_stop_failure_is_explicit_in_failure_record(self):
        protocol = compile_text(DEFAULT_TEXT)
        policy, _ = train(protocol, episodes=80, seed=17)
        driver = SimLabDriver(protocol, 6.0, 9.0, 20, np.random.default_rng(9),
                              bias=0.03)
        try:
            def fail_measure(_kind):
                raise RuntimeError("仪器失联")
            def fail_abort(_reason):
                raise RuntimeError("停机确认超时")
            driver.measure_ph = fail_measure
            driver.abort = fail_abort
            row = run_attempt(protocol, (6.0, 9.0, 20), policy, driver, index=1)
            self.assertFalse(row["safe_stop_confirmed"])
            self.assertIn("停机确认超时", row["abort_error"])
        finally:
            driver.close()

    def test_synthetic_calibration_has_held_out_validation(self):
        protocol = compile_text(DEFAULT_TEXT)
        rows = generate_synthetic_rows(protocol, count=80, seed=111)
        model, metrics = fit_model(rows, protocol, seed=112)
        self.assertEqual(metrics["validation_samples"], 16)
        self.assertLess(metrics["validation_rmse_ph"], 0.02)
        self.assertAlmostEqual(model.intercept, 4.795, delta=0.02)

    def test_motion_policy_covers_station_pairs(self):
        model, metrics = train_motion(episodes=100, seed=7)
        self.assertGreaterEqual(metrics["station_pair_policy_success"], 0.9)
        for source in STATION_NODES:
            for destination in STATION_NODES:
                if source != destination:
                    route, fallback = plan(model, source, destination)
                    self.assertTrue(route)
                    self.assertEqual(route[-1], STATION_NODES[destination])
                    self.assertIsInstance(fallback, bool)

    def test_obstacle_rl_improves_over_warm_start(self):
        model, metrics = train_motion(episodes=350, seed=52)
        self.assertEqual(metrics["obstacle_cases"], 84)
        self.assertGreater(metrics["after_rl_direct_success"],
                           metrics["warm_start_direct_success"] + 0.5)
        route, _ = plan(model, "doser", "ph_meter", blocked=BLOCKS[2])
        self.assertNotIn(BLOCKS[2], route)

    def test_policy_comparison_reuses_initial_conditions(self):
        protocol = compile_text(DEFAULT_TEXT)
        policy, _ = train(protocol, episodes=80, seed=23)
        evaluation = evaluate_dose_policies(protocol, policy, BufferModel(),
                                            seed=44, trials=20)
        policies = evaluation["policies"]
        self.assertEqual({row["trials"] for row in policies.values()}, {20})
        self.assertEqual({row["mean_initial_abs_error_ph"] for row in policies.values()},
                         {policies["dqn"]["mean_initial_abs_error_ph"]})
        self.assertEqual(policies["no_adjustment"]["mean_added_ml"], 0)


if __name__ == "__main__":
    unittest.main()
