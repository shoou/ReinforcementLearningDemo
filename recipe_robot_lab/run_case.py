"""End-to-end buffer formulation and embodied laboratory simulation."""
from __future__ import annotations

import argparse
import hashlib
import html
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

from calibration import calibrate
from compiler import DEFAULT_TEXT, Protocol, compile_actions, compile_text
from dose_rl import ACTIONS, action_mask, choose, observe, q_values, train
from driver import DeviceFault, LabDriver, SimLabDriver
from evaluation import evaluate_dose_policies, evaluate_suite
from motion_rl import train as train_motion


ROOT = Path(__file__).parent


def write_environment_manifest(args, output: Path) -> dict:
    packages = {}
    for package in ("numpy", "scipy", "torch", "Pillow", "pybullet",
                    "pybullet-arm64", "mujoco"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    digest = hashlib.sha256()
    source_paths = [*ROOT.glob("*.py"), ROOT / "run.sh",
                    *ROOT.glob("requirements*.txt")]
    for path in sorted(source_paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    manifest = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages,
        "source_sha256": digest.hexdigest(),
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
    }
    (output / "environment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def score_result(result: dict, protocol: Protocol) -> float:
    """Lower is better; pH accuracy dominates the demonstration objective."""
    ph_error = abs(result["final_ph"] - protocol.target_ph)
    stability_error = abs(result["stability_delta_ph"])
    stock_ml = result["actual_ml"]["citric_acid"] + result["actual_ml"]["trisodium_citrate"]
    return round(ph_error + stability_error + 0.003 * stock_ml
                 + 0.0003 * (result["elapsed_s"] - 600)
                 + 0.006 * len(result["adjustments"]), 5)


def experiment(protocol: Protocol, candidate: tuple[float, float, int], policy,
               driver: LabDriver, index: int,
               capture_path: Path | None = None) -> dict:
    acid, base, mix_seconds = candidate
    actions = compile_actions(protocol, acid, base, mix_seconds)
    start_trace = len(driver.trace)
    readings, adjustments, events = [], [], []
    adjustment_budget_ml = protocol.reserve_ml
    captured = False

    def event(name: str, **values):
        events.append({"step": len(events) + 1, "time_s": driver.state()["time_s"],
                       "event": name, **values})

    for command in actions:
        op = command["op"]
        if op == "identify":
            verified = driver.identify(command["object"])
            if not verified:
                raise RuntimeError("样品瓶身份核对失败")
            event("identify", object=command["object"], status="verified")
        elif op == "grasp":
            driver.grasp()
            event("grasp", object=command["object"])
        elif op == "move":
            driver.move(command["station"])
            event("move", station=command["station"])
            if capture_path and command["station"] == "ph_meter" and not captured:
                driver.capture(capture_path)
                captured = True
        elif op == "dispense":
            source = command["source"]
            actual = driver.dispense(source, command["volume_ml"])
            event("dispense", source=source, requested_ml=command["volume_ml"],
                  actual_ml=actual)
        elif op == "mix":
            driver.mix(mix_seconds)
            event("mix", seconds=mix_seconds)
        elif op == "measure":
            measurement = driver.measure_ph(command["property"])
            readings.append({"kind": command["property"], **measurement})
            event("measure", kind=command["property"], **measurement)
        elif op == "adjust":
            for _ in range(4):
                ph = readings[-1]["ph"]
                current = driver.state()
                available_ml = min(current["reserve_ml"], adjustment_budget_ml)
                obs = observe(ph, protocol, available_ml,
                              current["acid_base_ratio"])
                mask = action_mask(available_ml)
                action_id = choose(policy, obs, mask)
                event("rl_decision", observed_ph=ph,
                      reserve_ml=round(available_ml, 4),
                      action_index=action_id,
                      action=ACTIONS[action_id][0], amount_ml=ACTIONS[action_id][1],
                      q_values=q_values(policy, obs), legal_actions=mask.tolist())
                reagent, amount = ACTIONS[action_id]
                if reagent == "stop":
                    event("rl_stop", ph=ph)
                    break
                driver.move("doser")
                actual = driver.dispense(reagent, amount)
                adjustment_budget_ml -= amount
                driver.move("mixer")
                driver.mix(mix_seconds)
                driver.move("ph_meter")
                measurement = driver.measure_ph("adjusted_ph")
                updated = measurement["ph"]
                readings.append({"kind": "adjusted_ph", **measurement})
                adjustments.append({"reagent": reagent, "requested_ml": amount,
                                    "actual_ml": actual,
                                    "ph_before": ph, "ph_after": updated})
                event("rl_adjust", reagent=reagent, requested_ml=amount,
                      actual_ml=actual, ph_after=updated)
        elif op == "top_up":
            actual = driver.top_up(protocol.final_volume_ml)
            event("top_up", water_ml=actual)
        elif op == "hold":
            driver.hold(command["minutes"])
            event("hold", minutes=command["minutes"])
        elif op == "release":
            driver.release()
            event("release", object=command["object"])
        else:
            raise RuntimeError(f"未知协议操作: {op}")
    final_state = driver.state()
    final_ph = next(row["ph"] for row in readings if row["kind"] == "final_ph")
    stability_ph = next(row["ph"] for row in readings if row["kind"] == "ph_after_10min")
    result = {
        "experiment_id": f"SIM-{index:03d}", "mode": driver.mode, "status": "completed",
        "candidate": {"citric_acid_ml": acid, "trisodium_citrate_ml": base,
                      "mix_seconds": mix_seconds},
        "compiled_actions": actions,
        "initial_ph": readings[0]["ph"], "final_ph": final_ph,
        "ph_after_10min": stability_ph,
        "stability_delta_ph": round(stability_ph - final_ph, 4),
        "target_ph": protocol.target_ph, "mix_seconds": mix_seconds,
        "actual_ml": final_state["actual_ml"],
        "final_volume_ml": final_state["total_volume_ml"],
        "elapsed_s": final_state["time_s"],
        "unsafe_station_contacts": final_state["unsafe_station_contacts"],
        "motion_policy_moves": final_state["motion_policy_moves"],
        "motion_fallbacks": final_state["motion_fallbacks"],
        "adjustments": adjustments, "readings": readings,
        "events": events, "arm_trace": driver.trace[start_trace:],
        "sim_randomization": driver.randomization,
        "max_arm_position_error_m": max(
            (row.get("position_error_m", 0) for row in driver.trace[start_trace:]),
            default=0),
    }
    result["score"] = score_result(result, protocol)
    result["within_tolerance"] = abs(result["final_ph"] - protocol.target_ph) <= protocol.ph_tolerance
    result["stable_within_tolerance"] = (
        abs(result["ph_after_10min"] - protocol.target_ph) <= protocol.ph_tolerance
    )
    return result


def run_attempt(protocol: Protocol, candidate: tuple[float, float, int], policy,
                driver: LabDriver, index: int,
                capture_path: Path | None = None) -> dict:
    """Always stop and record the workcell when protocol execution fails."""
    try:
        return experiment(protocol, candidate, policy, driver, index, capture_path)
    except Exception as exc:
        abort_error = None
        try:
            driver.abort(f"{type(exc).__name__}: {exc}")
        except Exception as stop_exc:
            abort_error = f"{type(stop_exc).__name__}: {stop_exc}"
        capture_error = None
        if capture_path is not None:
            try:
                driver.capture(capture_path)
            except Exception as image_exc:
                capture_error = f"{type(image_exc).__name__}: {image_exc}"
        try:
            state = driver.state()
        except Exception as state_exc:
            state = {"unavailable": f"{type(state_exc).__name__}: {state_exc}"}
        fault = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, DeviceFault):
            fault.update(operation=exc.operation, occurrence=exc.occurrence)
        return {
            "experiment_id": f"SIM-{index:03d}", "mode": driver.mode,
            "status": "failed", "fault": fault,
            "candidate": {"citric_acid_ml": candidate[0],
                          "trisodium_citrate_ml": candidate[1],
                          "mix_seconds": candidate[2]},
            "safe_stop_confirmed": abort_error is None,
            "abort_error": abort_error, "capture_error": capture_error,
            "state_at_abort": state, "arm_trace": list(driver.trace),
            "sim_randomization": driver.randomization,
        }


def candidates(protocol: Protocol) -> list[tuple[float, float, int]]:
    scale = protocol.final_volume_ml / 20
    choices = []
    for acid_base in np.arange(4.0, 8.1, 0.5):
        for base_base in np.arange(6.0, 11.1, 0.5):
            acid, base = round(float(acid_base * scale), 3), round(float(base_base * scale), 3)
            if acid + base > protocol.final_volume_ml - protocol.reserve_ml:
                continue
            # Teaching constraint: at least half of the lower stock concentration.
            total_mmol = acid * protocol.acid_stock_m + base * protocol.base_stock_m
            if total_mmol / protocol.final_volume_ml < 0.5 * min(protocol.acid_stock_m, protocol.base_stock_m):
                continue
            for mix in sorted({10, 20, 30, protocol.mix_seconds}):
                choices.append((acid, base, mix))
    return choices


def propose(protocol: Protocol, history: list[dict], available: list[tuple]) -> tuple:
    """Expected improvement over a GP fit to measured experiment scores."""
    if len(history) < 3:
        raise ValueError("贝叶斯优化至少需要三次初始试验")
    scale = protocol.final_volume_ml / 20
    mix_scale = max(30, protocol.mix_seconds)

    def feature(c):
        return [c[0] / scale / 8, c[1] / scale / 11, c[2] / mix_scale]

    x = np.array([feature((r["candidate"]["citric_acid_ml"],
                           r["candidate"]["trisodium_citrate_ml"],
                           r["candidate"]["mix_seconds"])) for r in history])
    y = np.array([r["score"] for r in history])
    # Fixed-scale RBF Gaussian process; Cholesky keeps small data fits stable.
    trial_x = np.array([feature(c) for c in available])
    length = np.array([0.4, 0.4, 0.5])
    amplitude = 0.25

    def covariance(a, b):
        distance2 = np.sum(((a[:, None, :] - b[None, :, :]) / length) ** 2, axis=2)
        return amplitude ** 2 * np.exp(-0.5 * distance2)

    k = covariance(x, x) + np.eye(len(x)) * 0.02 ** 2
    chol = np.linalg.cholesky(k)
    centered = y - float(np.mean(y))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
    cross = covariance(trial_x, x)
    mean = float(np.mean(y)) + np.sum(cross * alpha[None, :], axis=1)
    projection = np.linalg.solve(chol, cross.T)
    std = np.sqrt(np.maximum(1e-10, amplitude ** 2 - np.sum(projection ** 2, axis=0)))
    improvement = y.min() - mean - 0.002
    z = improvement / np.maximum(std, 1e-9)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return available[int(np.argmax(ei))]


def curve_svg(points: list[dict], field: str, colour: str) -> str:
    """Small dependency-free training chart with a fixed 0–100% y scale."""
    if not points:
        return ""
    width, height = 430, 170
    last = max(point["episode"] for point in points)
    coords = [(30 + 385 * point["episode"] / last,
               140 - 115 * point[field]) for point in points]
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(coords))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="训练成功率曲线">'
            f'<path d="M30,25 V140 H415" stroke="#a9bcb8" fill="none"/>'
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="3"/>'
            f'<text x="2" y="30" font-size="11">100%</text>'
            f'<text x="8" y="143" font-size="11">0%</text>'
            f'<text x="385" y="160" font-size="11">{last} 回合</text></svg>')


def motion_route_svg(example: dict) -> str:
    """Show one obstacle-aware route in the discrete planning workspace."""
    from motion_rl import STATION_NODES

    def xy(node):
        return 42 + 82 * node[0], 290 - 40 * node[1]

    start = STATION_NODES[example["source"]]
    goal = STATION_NODES[example["destination"]]
    nodes = [start, *example["route"]]
    path = " ".join(f"{xy(node)[0]},{xy(node)[1]}" for node in nodes)
    grid = "".join(f'<circle cx="{42 + 82 * x}" cy="{290 - 40 * y}" r="3" '
                   f'fill="#c1cfca"/>' for x in range(3) for y in range(7))
    bx, by = xy(example["blocked_node"])
    sx, sy = xy(start)
    gx, gy = xy(goal)
    return (f'<svg viewBox="0 0 255 330" role="img" '
            f'aria-label="有禁行路点的机械臂规划路线">{grid}'
            f'<polyline points="{path}" fill="none" stroke="#0d9488" '
            f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<rect x="{bx - 11}" y="{by - 11}" width="22" height="22" '
            f'rx="3" fill="#e05c54"/>'
            f'<circle cx="{sx}" cy="{sy}" r="9" fill="#3678bf"/>'
            f'<circle cx="{gx}" cy="{gy}" r="9" fill="#e8a432"/>'
            f'<text x="3" y="318" font-size="12">蓝:起点 · 金:终点 · 红:禁行</text></svg>')


def write_report(output: Path, protocol: Protocol, history: list[dict],
                 training: dict, next_candidate: tuple, failed_count: int,
                 calibration: dict, motion_training: dict, validation: dict,
                 evaluation: dict, stress_evaluation: dict,
                 mujoco_audit: dict | None, replay_meta: dict | None):
    best = min(history, key=lambda r: r["score"])
    width, height = 760, 210
    pad = 36
    max_error = max(0.25, max(abs(r["final_ph"] - protocol.target_ph) for r in history))
    points = []
    for i, row in enumerate(history):
        x = pad + i * (width - 2 * pad) / max(1, len(history) - 1)
        y = height - pad - abs(row["final_ph"] - protocol.target_ph) / max_error * (height - 2 * pad)
        points.append((x, y))
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#12a8a2" />'
                      for x, y in points)
    rows = "\n".join(
        f"<tr><td>{r['experiment_id']}</td><td>{r['candidate']['citric_acid_ml']:.2f} / "
        f"{r['candidate']['trisodium_citrate_ml']:.2f}</td><td>{r['mix_seconds']}</td>"
        f"<td>{r['initial_ph']:.3f}</td><td>{r['final_ph']:.3f}</td>"
        f"<td>{r['ph_after_10min']:.3f}</td><td>{len(r['adjustments'])}</td>"
        f"<td>{r['score']:.3f}</td></tr>" for r in history)
    protocol_text = html.escape(json.dumps(protocol.as_dict(), ensure_ascii=False, indent=2))
    action_text = html.escape(json.dumps(best["events"], ensure_ascii=False, indent=2))
    dose_curve = curve_svg(training["curve"], "success_rate", "#0d9488")
    motion_curve = curve_svg(motion_training["curve"], "success_rate", "#d97706")
    obstacle_route = motion_route_svg(motion_training["example_obstacle_route"])
    policy_rows = "".join(
        f"<tr><td>{label}</td><td>{evaluation['policies'][name]['tolerance_hit_rate']:.1%} "
        f"({evaluation['policies'][name]['hit_rate_ci95'][0]:.1%}–"
        f"{evaluation['policies'][name]['hit_rate_ci95'][1]:.1%})</td>"
        f"<td>{evaluation['policies'][name]['mean_final_abs_error_ph']:.4f}</td>"
        f"<td>{evaluation['policies'][name]['mean_added_ml']:.3f}</td></tr>"
        for name, label in (("dqn", "补液 DQN"), ("fixed_step", "固定步长规则"),
                            ("model_based", "模型一步预测"),
                            ("no_adjustment", "不调整")))
    stress_rows = "".join(
        f'<tr><td>{label}</td>' + ''.join(
            f'<td>{stress_evaluation["scenarios"][scenario]["aggregate"][name]["mean_hit_rate"]:.1%}'
            f' <small class="muted">[{stress_evaluation["scenarios"][scenario]["aggregate"][name]["min_hit_rate"]:.1%}–'
            f'{stress_evaluation["scenarios"][scenario]["aggregate"][name]["max_hit_rate"]:.1%}]</small></td>'
            for scenario in ("in_distribution", "recipe_edge", "model_shift", "combined_shift"))
        + '</tr>' for name, label in (("dqn", "补液 DQN"),
                                      ("model_based", "模型一步预测"),
                                      ("fixed_step", "固定步长")))
    replay_data = json.dumps([
        {"id": row["experiment_id"], "events": row["events"],
         "candidate": row["candidate"], "score": row["score"]}
        for row in history], ensure_ascii=False).replace("<", "\\u003c")
    replay_options = "".join(f'<option value="{i}">{r["experiment_id"]} · '
                             f'{r["final_ph"]:.3f} pH</option>' for i, r in enumerate(history))
    mujoco_html = ""
    if mujoco_audit:
        dynamic = mujoco_audit["dynamic_replay"]
        mujoco_html = (
            f'<section><h2>MuJoCo 跨引擎动力学回放</h2><p>静态复核 '
            f'{mujoco_audit["checked_poses"]} 个姿态，末端最大差异 '
            f'{mujoco_audit["max_marker_disagreement_m"] * 1000:.3f} mm；'
            f'带固定 80 g 负载重放 {dynamic["replayed_waypoints"]} 个路点，'
            f'末端最大误差 {dynamic["max_waypoint_error_m"] * 1000:.2f} mm，'
            f'机械臂与工位穿透采样 {dynamic["station_penetration_samples"]} 次。'
            f'关节速度峰值 {dynamic["max_abs_joint_speed_rad_s"]:.2f} rad/s，'
            f'控制目标斜率上限 {dynamic["command_slew_limit_rad_s"]:.2f} rad/s，'
            f'执行器力矩到限采样占比 {dynamic["effort_saturation_fraction"]:.1%}。'
            f'其中第 7 轴到限占比 {dynamic["joint_saturation_fractions"][6]:.1%}。'
            f'样品瓶碰撞在回放中关闭，且没有真实夹爪力或液体模型；'
            f'力矩到限需在真机迁移前处理。'
            f'<a href="mujoco_audit.json">查看完整回放指标</a>。</p></section>'
        )
    replay_html = (f'<section><h2>机械臂动作动画</h2><img src="motion_replay.gif" '
                   f'alt="根据最佳实验记录的关节姿态制作的机械臂动作动画" '
                   f'style="width:100%;border-radius:8px"><p class="muted">'
                   f'{replay_meta["frames"]} 帧记录姿态可视化；未重新执行物理实验。</p></section>') \
                  if replay_meta else ""
    scene_html = ('<section><h2>PyBullet 机械臂工作站</h2><img src="scene.png" '
                  'alt="Panda 机械臂将样品瓶送至测量工位的仿真画面" '
                  'style="width:100%;border-radius:8px"></section>') \
                 if (output / "scene.png").exists() else ""
    validation_html = (
        f'{validation["repeats"]} / {validation["attempted_repeats"]} 次模拟重复完成，'
        f'平均终测 pH {validation["mean_final_ph"]:.3f}，'
        f'标准差 {validation["std_final_ph"]:.4f}；'
        f'10 分钟后仍在目标容差内 {validation["stable_hit_rate"]:.1%}。'
        if validation["repeats"] else
        f'0 / {validation["attempted_repeats"]} 次模拟重复完成；故障记录见 validation.jsonl。'
    )
    proposal_title = "下一轮建议" if len(history) >= 3 else "下一轮初始试验"
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>配方编译 × 具身实验闭环</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#21343a;background:#f3f6f4;margin:0}}
main{{max-width:980px;margin:auto;padding:40px 22px}}h1{{font-size:30px;margin-bottom:4px}}
.sub{{color:#577078}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card,section{{background:white;border:1px solid #dae5e0;border-radius:14px;padding:22px;margin:18px 0}}
.card,section{{overflow-x:auto}}
.card b{{display:block;font-size:28px;color:#086f6a}}table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #e5eeea}}th{{color:#4d686c}}
pre{{overflow:auto;background:#f5f8f7;padding:15px;border-radius:8px}}svg{{max-width:100%}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.muted{{color:#637b80}}
.timeline{{display:grid;gap:8px;margin-top:15px}}.timeline div{{border-left:3px solid #12a8a2;padding:5px 12px;background:#f5f8f7}}
select{{padding:10px;border:1px solid #bed3ce;border-radius:8px;font:inherit}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}table{{font-size:13px}}}}
</style><main><h1>配方编译 × 具身实验闭环</h1>
<p class="sub">全部结果均为仿真数据；pH 使用教学代理模型，设备命令为模拟应答。</p>
<div class="grid"><div class="card">成功 / 故障<b>{len(history)} / {failed_count}</b></div>
<div class="card">最佳 pH<b>{best['final_ph']:.3f}</b></div>
<div class="card">最佳评分<b>{best['score']:.3f}</b></div></div>
<section><h2>目标 pH 误差</h2><svg viewBox="0 0 {width} {height}" role="img" aria-label="各轮实验的目标 pH 误差折线图">
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9bb7b2"/>
<path d="{path}" fill="none" stroke="#12a8a2" stroke-width="3"/>{circles}</svg></section>
{scene_html}
{replay_html}
<section><h2>实验记录</h2><table><thead><tr><th>编号</th><th>酸 / 盐母液 mL</th><th>混合 s</th>
<th>初测 pH</th><th>终测 pH</th><th>10 分钟 pH</th><th>RL 调整</th><th>评分 ↓</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>强化学习训练</h2><div class="charts"><div><h3>补液 DQN</h3>{dose_curve}</div>
<div><h3>机械臂路点 DQN</h3>{motion_curve}</div></div><p class="muted">曲线为每 50 回合成功率；
运动策略的 12 条工位路线平均 {motion_training['mean_route_steps']:.2f} 步，
图搜索最短路线平均 {motion_training['mean_shortest_steps']:.2f} 步。
动态禁行路点的 {motion_training['obstacle_cases']} 组挑战中，示范预热直接完成率
{motion_training['warm_start_direct_success']:.1%}，RL 更新后
{motion_training['after_rl_direct_success']:.1%}；图搜索参照可完成全部规划挑战，
策略失败时由图搜索回退。
这些禁行路点属于规划任务，未作为实体障碍放入主工作站。</p>
<div style="max-width:300px;margin:auto">{obstacle_route}</div></section>
<section><h2>补液策略同条件对比</h2><p>在训练外生成的 {evaluation['policies']['dqn']['trials']} 组
合成状态上，三种策略使用相同初始配方与仪器偏差。</p>
<table><thead><tr><th>策略</th><th>目标容差命中率</th><th>平均最终 pH 误差</th>
<th>平均补液 mL</th></tr></thead><tbody>{policy_rows}</tbody></table></section>
<section><h2>分布变化压力评估</h2><p>{stress_evaluation['training_seeds']} 个独立 DQN 训练种子，
每种场景各 {stress_evaluation['trials_per_scenario']} 组配对试验。以下为平均命中率，
方括号是训练种子之间的范围；
完整分种子结果见 policy_evaluation.json。</p>
<table><thead><tr><th>比较</th><th>训练分布</th><th>配方边界</th>
<th>模型及仪器偏移</th><th>组合偏移</th></tr></thead><tbody>{stress_rows}</tbody></table></section>
<section><h2>逐轮动作回放</h2><label for="replay">选择实验：</label>
<select id="replay">{replay_options}</select><p id="replay-summary"></p>
<div id="replay-events" class="timeline"></div></section>
<section><h2>编译协议</h2><pre>{protocol_text}</pre></section>
<section><h2>模拟数据标定</h2><p>训练样本 {calibration['train_samples']}，独立验证样本 {calibration['validation_samples']}，
验证集 pH 均方根误差 {calibration['validation_rmse_ph']:.4f}。这些数据由代理模型生成，不能证明真实化学预测精度。</p></section>
<section><h2>最佳候选独立重复</h2><p>{validation_html}</p></section>
<section><h2>最佳实验动作与记录</h2><pre>{action_text}</pre></section>
{mujoco_html}
<section><h2>{proposal_title}</h2><p>柠檬酸母液 {next_candidate[0]:.2f} mL，柠檬酸钠母液 {next_candidate[1]:.2f} mL，
混合 {next_candidate[2]} 秒。该候选仅由仿真数据提出，尚未执行。</p></section>
<section><h2>数据与复现</h2><p><a href="experiments.jsonl">逐轮实验日志</a> ·
<a href="validation.jsonl">重复试验</a> ·
<a href="policy_evaluation.json">策略评估</a> ·
<a href="environment.json">环境与源码指纹</a></p></section>
<p class="sub">DQN 训练：{training['episodes']} 回合；最近 100 回合命中率 {training['training_success_last_100']:.1%}。
机械臂工位运动 DQN：{motion_training['episodes']} 回合；12 组工位路线中策略直接完成比例
 {motion_training['station_pair_policy_success']:.1%}。完整传感器读数和轨迹见 experiments.jsonl。</p>
<script>const runs={replay_data};const selector=document.getElementById('replay');
function showRun(){{const run=runs[Number(selector.value)];
document.getElementById('replay-summary').textContent=`酸 ${{run.candidate.citric_acid_ml}} mL / 盐 ${{run.candidate.trisodium_citrate_ml}} mL · 混合 ${{run.candidate.mix_seconds}} s · 评分 ${{run.score}}`;
const box=document.getElementById('replay-events');box.replaceChildren();
for(const step of run.events){{const item=document.createElement('div');
let detail=step.station||step.source||step.kind||step.action||step.object||'';
if(step.event==='rl_decision') detail+=` · ${{step.amount_ml}} mL · Q=[${{step.q_values.join(', ')}}]`;
if(step.ph!==undefined) detail+=` · pH ${{step.ph}}`;
item.textContent=`${{step.step}} · ${{step.time_s}} s · ${{step.event}} · ${{detail}}`;
box.appendChild(item);}}}}
selector.addEventListener('change',showRun);showRun();</script></main></html>"""
    (output / "report.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="中文实验说明")
    parser.add_argument("--rounds", type=int, default=6, help="实验总次数，至少 3")
    parser.add_argument("--rl-episodes", type=int, default=900)
    parser.add_argument("--motion-episodes", type=int, default=350)
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--eval-training-seeds", type=int, default=3,
                        help="跨独立 DQN 训练种子的压力评估次数")
    parser.add_argument("--eval-trials", type=int, default=120,
                        help="每种压力场景的配对合成试验数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--gui", action="store_true", help="显示 PyBullet 窗口")
    parser.add_argument("--fault-op", help="首轮注入一次模拟故障，例如 measure:1")
    parser.add_argument("--mujoco-model", type=Path,
                        help="可选：官方 MuJoCo Menagerie Panda MJCF 的 panda.xml 路径")
    args = parser.parse_args()
    if not 3 <= args.rounds <= 30:
        parser.error("--rounds 须在 3–30 之间")
    if not 1 <= args.validation_repeats <= 20:
        parser.error("--validation-repeats 须在 1–20 之间")
    if args.rl_episodes < 80 or args.motion_episodes < 80:
        parser.error("两种 DQN 的训练回合数均须至少为 80")
    if not 1 <= args.eval_training_seeds <= 5 or not 20 <= args.eval_trials <= 1000:
        parser.error("评估训练种子须在 1–5，每场景试验数须在 20–1000")
    fault_at = None
    if args.fault_op:
        parts = args.fault_op.split(":")
        if len(parts) != 2 or parts[0] not in {"dispense", "mix", "measure", "top_up", "hold"}:
            parser.error("--fault-op 格式为 dispense|mix|measure|top_up|hold:正整数")
        try:
            fault_at = (parts[0], int(parts[1]))
        except ValueError:
            parser.error("--fault-op 调用次数必须是正整数")
        if fault_at[1] < 1:
            parser.error("--fault-op 调用次数必须是正整数")
    protocol = compile_text(args.text)
    args.output.mkdir(parents=True, exist_ok=True)
    for artifact in ("scene.png", "mujoco_audit.json", "motion_replay.gif"):
        (args.output / artifact).unlink(missing_ok=True)
    for old_policy in args.output.glob("dose_dqn_seed_*.pt"):
        old_policy.unlink()
    write_environment_manifest(args, args.output)
    (args.output / "protocol.json").write_text(
        json.dumps({"source_text": args.text, "protocol": protocol.as_dict()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    buffer_model, calibration_metrics = calibrate(protocol, args.output, args.seed + 500)
    policy, training = train(protocol, args.rl_episodes, args.seed,
                             model=buffer_model)
    motion_policy, motion_training = train_motion(args.motion_episodes, args.seed + 10)
    evaluation = evaluate_dose_policies(protocol, policy, buffer_model,
                                        args.seed + 50000, trials=args.eval_trials)
    stress_policies = [policy]
    training_seed_values = [args.seed]
    for seed_index in range(1, args.eval_training_seeds):
        training_seed = args.seed + seed_index * 1009
        repeat_policy, _ = train(protocol, args.rl_episodes,
                                 training_seed, model=buffer_model)
        stress_policies.append(repeat_policy)
        training_seed_values.append(training_seed)
    stress_evaluation = evaluate_suite(protocol, stress_policies, buffer_model,
                                       args.seed + 50000, trials=args.eval_trials)
    stress_evaluation["training_seed_values"] = training_seed_values
    (args.output / "policy_evaluation.json").write_text(
        json.dumps({"paired_default": evaluation, "stress": stress_evaluation},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    torch_path = args.output / "dose_dqn.pt"
    import torch
    torch.save(policy.state_dict(), torch_path)
    for seed_value, trained_policy in zip(training_seed_values[1:], stress_policies[1:]):
        torch.save(trained_policy.state_dict(),
                   args.output / f"dose_dqn_seed_{seed_value}.pt")
    torch.save(motion_policy.state_dict(), args.output / "motion_dqn.pt")
    rng = np.random.default_rng(args.seed + 1000)
    available = candidates(protocol)
    scale = protocol.final_volume_ml / 20
    initial = [(8.0 * scale, 6.0 * scale, protocol.mix_seconds),
               (7.0 * scale, 8.0 * scale, protocol.mix_seconds),
               (6.0 * scale, 9.0 * scale, protocol.mix_seconds)]
    history, failed = [], []
    with (args.output / "experiments.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(args.rounds):
            if len(history) < 3:
                candidate = next((c for c in initial[len(history):] if c in available), available[0])
            else:
                candidate = propose(protocol, history, available)
            available = [c for c in available if c != candidate]
            driver = SimLabDriver(protocol, candidate[0], candidate[1], candidate[2],
                                  rng, bias=0.0, gui=args.gui,
                                  fault_at=fault_at if i == 0 else None,
                                  model=buffer_model, motion_policy=motion_policy)
            try:
                row = run_attempt(protocol, candidate, policy, driver, index=i + 1,
                                  capture_path=args.output / "scene.png" if i == 0 else None)
            finally:
                driver.close()
            row["selection_method"] = "initial_design" if len(history) < 3 else "gp_expected_improvement"
            if row["status"] == "completed":
                history.append(row)
            else:
                failed.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row["status"] == "failed":
                print(f"{row['experiment_id']}: {row['fault']['type']}，"
                      f"安全停止={'完成' if row['safe_stop_confirmed'] else '失败'}，已记录")
                if not row["safe_stop_confirmed"]:
                    print("安全停止失败，终止后续实验")
                    break
                continue
            print(f"{row['experiment_id']}: pH {row['initial_ph']:.3f} → "
                  f"{row['final_ph']:.3f}, score {row['score']:.3f}, "
                  f"RL 调整 {len(row['adjustments'])} 次")
    if not history:
        failure_summary = {"mode": "simulation", "status": "no_successful_experiments",
                           "total_attempts": len(failed),
                           "successful_experiments": 0,
                           "failed_experiments": len(failed),
                           "faults": [{"experiment_id": row["experiment_id"],
                                       "fault": row["fault"],
                                       "safe_stop_confirmed": row["safe_stop_confirmed"]}
                                      for row in failed]}
        (args.output / "summary.json").write_text(
            json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "report.html").write_text(
            "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<title>实验失败记录</title><h1>没有成功的模拟实验</h1>"
            "<p>故障与安全停止状态见 summary.json 和 experiments.jsonl。</p>",
            encoding="utf-8")
        raise SystemExit("所有模拟实验均失败；故障记录已保存")
    best = min(history, key=lambda r: r["score"])
    from render_replay import render_replay
    replay_error = None
    try:
        replay_meta = render_replay(best, args.output / "motion_replay.gif")
    except Exception as exc:
        replay_meta = None
        replay_error = f"{type(exc).__name__}: {exc}"
    best_tuple = (best["candidate"]["citric_acid_ml"],
                  best["candidate"]["trisodium_citrate_ml"],
                  best["candidate"]["mix_seconds"])
    validation_rng = np.random.default_rng(args.seed + 20000)
    validation_rows = []
    with (args.output / "validation.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(args.validation_repeats):
            validation_driver = SimLabDriver(
                protocol, *best_tuple, validation_rng,
                bias=float(validation_rng.normal(0.0, 0.005)),
                model=buffer_model, motion_policy=motion_policy, gui=args.gui)
            try:
                validation_row = run_attempt(protocol, best_tuple, policy,
                                             validation_driver, index=1000 + index)
            finally:
                validation_driver.close()
            validation_row["selection_method"] = "independent_repeat"
            validation_rows.append(validation_row)
            stream.write(json.dumps(validation_row, ensure_ascii=False) + "\n")
            if validation_row["status"] == "failed" and not validation_row["safe_stop_confirmed"]:
                break
    completed_validation = [row for row in validation_rows if row["status"] == "completed"]
    validation_ph = np.array([row["final_ph"] for row in completed_validation])
    validation_metrics = {
        "source": "simulation", "repeats": len(completed_validation),
        "attempted_repeats": len(validation_rows),
        "failed_repeats": len(validation_rows) - len(completed_validation),
        "mean_final_ph": round(float(np.mean(validation_ph)), 4)
        if len(validation_ph) else None,
        "std_final_ph": round(float(np.std(validation_ph, ddof=1)), 4)
        if len(validation_ph) > 1 else (0.0 if len(validation_ph) else None),
        "stable_hit_rate": round(float(np.mean(
            [row["stable_within_tolerance"] for row in completed_validation])), 3)
        if completed_validation else None,
        "mean_abs_10min_drift_ph": round(float(np.mean(
            [abs(row["stability_delta_ph"]) for row in completed_validation])), 4)
        if completed_validation else None,
    }
    mujoco_audit = None
    if args.mujoco_model:
        from mujoco_check import audit
        mujoco_audit = audit(args.mujoco_model, args.output / "experiments.jsonl")
        (args.output / "mujoco_audit.json").write_text(
            json.dumps(mujoco_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        if (not mujoco_audit["within_5mm"] or
                not mujoco_audit["dynamic_replay"]["within_25mm_and_no_station_penetration"]):
            raise RuntimeError("MuJoCo 运动学或动力学回放超出阈值")
    next_candidate = propose(protocol, history, available) if len(history) >= 3 else available[0]
    summary = {"mode": "simulation", "protocol": protocol.as_dict(),
               "calibration": calibration_metrics,
               "policy_evaluation": evaluation,
               "stress_evaluation": stress_evaluation,
               "mujoco_audit": mujoco_audit,
               "replay": replay_meta,
               "replay_error": replay_error,
               "environment_manifest": "environment.json",
               "best_candidate_validation": validation_metrics,
               "total_attempts": len(history) + len(failed),
               "successful_experiments": len(history),
               "failed_experiments": len(failed), "seed": args.seed,
               "rl_training": training, "best_experiment_id": best["experiment_id"],
               "motion_rl_training": motion_training,
               "best_candidate": best["candidate"], "best_final_ph": best["final_ph"],
               "first_score": history[0]["score"], "best_score": best["score"],
               "next_proposal": {"citric_acid_ml": next_candidate[0],
                                 "trisodium_citrate_ml": next_candidate[1],
                                 "mix_seconds": next_candidate[2]},
               "tolerance_hit_rate": round(sum(r["within_tolerance"] for r in history) / len(history), 3),
               "stability_tolerance_hit_rate": round(
                   sum(r["stable_within_tolerance"] for r in history) / len(history), 3),
               "mean_abs_10min_drift_ph": round(
                   float(np.mean([abs(r["stability_delta_ph"]) for r in history])), 4),
               "all_arm_moves_within_25mm": all(r["max_arm_position_error_m"] <= 0.025
                                                for r in history),
               "all_without_station_contacts": all(r["unsafe_station_contacts"] == 0
                                                   for r in history),
               "motion_policy_moves": sum(r["motion_policy_moves"] for r in history),
               "motion_fallbacks": sum(r["motion_fallbacks"] for r in history)}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    write_report(args.output, protocol, history, training, next_candidate,
                 len(failed), calibration_metrics, motion_training, validation_metrics,
                 evaluation, stress_evaluation, mujoco_audit, replay_meta)
    print(f"完整记录：{args.output / 'report.html'}")


if __name__ == "__main__":
    main()
