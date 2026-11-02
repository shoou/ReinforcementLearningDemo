"""CPU-friendly 3D Panda sorting: Bullet geometry, Q-learning, motor replay.

The Q-table trains on a discrete graph whose nodes and edges are validated by
PyBullet IK and collision queries. Replay uses PyBullet motor control and physics.
This is a simulation prototype, not a real robot controller.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit("缺少 PyBullet；请按 README.md 安装 pybullet 或 pybullet-arm64") from exc


ROOT = Path(__file__).parent
X = (0.40, 0.48, 0.56, 0.64, 0.72)
Y = (-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24)
Z = (0.12, 0.20, 0.28, 0.36)
HOME = (2, 3, 3)
STATION_COLUMNS = (0, 1, 3, 4)
PICKS = tuple((i, 0, 0) for i in STATION_COLUMNS)
BINS = tuple((i, 6, 0) for i in STATION_COLUMNS)
DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (0, 0, 1), (0, 0, -1))
REST = (0.0, -0.4, 0.0, -2.3, 0.0, 2.0, 0.8)
OBSTACLE_CENTER = (0.56, 0.0, 0.115)
OBSTACLE_HALF = (0.20, 0.035, 0.115)
PARCEL_HALF = (0.025, 0.025, 0.025)
PARCEL_DROP = 0.08  # parcel centre is this far below grasp target
TOOL_LINK = 11
MAX_STEPS = 45
SIM_DT = 1 / 240


def xyz(node):
    return (X[node[0]], Y[node[1]], Z[node[2]])


def add_box(half, pos, rgba, mass=0.0, collision=True):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collision else -1
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                             baseVisualShapeIndex=visual, basePosition=pos)


class BulletCell:
    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("PyBullet connection failed")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_DT)
        p.loadURDF("plane.urdf", [0, 0, -0.005])
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.obstacle = add_box(OBSTACLE_HALF, OBSTACLE_CENTER, (0.83, 0.19, 0.17, 1))
        # Four indexed conveyor stops and four separate receiving surfaces.
        add_box((0.20, 0.05, 0.009), (0.56, -0.24, 0.001),
                (0.19, 0.43, 0.67, 1))
        bin_colours = ((0.20, 0.62, 0.36, 1), (0.25, 0.58, 0.72, 1),
                       (0.73, 0.51, 0.24, 1), (0.62, 0.36, 0.70, 1))
        for idx, bin_node in enumerate(BINS):
            x, y, _ = xyz(bin_node)
            add_box((0.036, 0.05, 0.009), (x, y, 0.001), bin_colours[idx])
            add_box((0.014, 0.003, 0.002), (x, y, 0.012),
                    (1, 0.9, 0.2, 1), collision=False)
        for pick_node in PICKS:
            x, y, _ = xyz(pick_node)
            add_box((0.004, 0.041, 0.002), (x, y, 0.012),
                    (0.8, 0.92, 1, 1), collision=False)
        self.lower, self.upper = [], []
        for joint in range(7):
            info = p.getJointInfo(self.robot, joint)
            self.lower.append(info[8])
            self.upper.append(info[9])
        self.poses = {}
        self.edges = {0: {}, 1: {}}
        self.parcel = None
        self.constraint = None
        self.reset_joints(REST)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def reset_joints(self, values):
        for joint, value in enumerate(values):
            p.resetJointState(self.robot, joint, value)
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                    targetPosition=value, force=87, maxVelocity=1.6)
        for joint in (9, 10):
            p.resetJointState(self.robot, joint, 0.04)

    def tool_position(self):
        return p.getLinkState(self.robot, TOOL_LINK, computeForwardKinematics=True)[4]

    def ik(self, point):
        # The nullspace solver depends on current joints. Start from one
        # consistent, joint-limit-compliant rest pose for graph construction.
        self.reset_joints(REST)
        q = p.calculateInverseKinematics(
            self.robot, TOOL_LINK, point, p.getQuaternionFromEuler((math.pi, 0, 0)),
            lowerLimits=self.lower, upperLimits=self.upper,
            jointRanges=[hi - lo for lo, hi in zip(self.lower, self.upper)],
            restPoses=REST, maxNumIterations=180, residualThreshold=1e-4)[:7]
        self.reset_joints(q)
        if math.dist(self.tool_position(), point) > 0.01:
            return None
        if any(v < lo - 1e-3 or v > hi + 1e-3
               for v, lo, hi in zip(q, self.lower, self.upper)):
            return None
        return tuple(q)

    def robot_clear(self, q):
        self.reset_joints(q)
        return not p.getClosestPoints(self.robot, self.obstacle, 0.012)

    def parcel_clear(self, point):
        # Conservative AABB check for the vacuum-held parcel. The margin is
        # applied in world coordinates; no claim of continuous swept volume.
        x, y, z = point
        ox, oy, oz = OBSTACLE_CENTER
        hx, hy, hz = OBSTACLE_HALF
        margin = 0.01
        return (abs(x - ox) > hx + PARCEL_HALF[0] + margin or
                abs(y - oy) > hy + PARCEL_HALF[1] + margin or
                abs((z - PARCEL_DROP) - oz) > hz + PARCEL_HALF[2] + margin)

    def edge_clear(self, a, b, carrying):
        if b not in self.poses:
            return False
        qa, qb = self.poses[a], self.poses[b]
        for step in range(11):
            f = step / 10
            q = [v + (w - v) * f for v, w in zip(qa, qb)]
            if not self.robot_clear(q):
                return False
            if carrying and not self.parcel_clear(self.tool_position()):
                return False
        return True

    def build_graph(self):
        for i in range(len(X)):
            for j in range(len(Y)):
                for k in range(len(Z)):
                    node = (i, j, k)
                    q = self.ik(xyz(node))
                    if q is not None and self.robot_clear(q):
                        self.poses[node] = q
        for carrying in (0, 1):
            for node in self.poses:
                legal = {}
                for action, delta in enumerate(DIRECTIONS):
                    nxt = tuple(a + b for a, b in zip(node, delta))
                    if self.edge_clear(node, nxt, carrying):
                        legal[action] = nxt
                self.edges[carrying][node] = legal
        required = (HOME, *PICKS, *BINS)
        if any(node not in self.poses for node in required):
            raise RuntimeError("home or work station is outside the safe IK graph")
        for pick in PICKS:
            for bin_node in BINS:
                if self.shortest(HOME, pick, 0) is None or self.shortest(pick, bin_node, 1) is None:
                    raise RuntimeError("a work order has no collision-free path")

    def shortest(self, start, goal, carrying):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, steps = queue.popleft()
            if node == goal:
                return steps
            for nxt in self.edges[carrying][node].values():
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, steps + 1))
        return None

    def new_parcel(self, pickup, offset=(0.0, 0.0), mass=0.08, friction=0.8):
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None
        if self.parcel is not None:
            p.removeBody(self.parcel)
        x, y, _ = xyz(PICKS[pickup])
        self.parcel = add_box(PARCEL_HALF, (x + offset[0], y + offset[1],
                                           PARCEL_HALF[2] + 0.012),
                              (1.0, 0.58, 0.11, 1), mass=mass)
        p.changeDynamics(self.parcel, -1, lateralFriction=friction)

    def attach(self):
        if math.dist(self.tool_position(), p.getBasePositionAndOrientation(self.parcel)[0]) > 0.10:
            return False
        self.constraint = p.createConstraint(
            self.robot, TOOL_LINK, self.parcel, -1, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, PARCEL_DROP], [0, 0, 0])
        p.changeConstraint(self.constraint, maxForce=80)
        for joint in (9, 10):
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                    targetPosition=0.005, force=20)
        return True

    def release(self):
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None
        for joint in (9, 10):
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                    targetPosition=0.04, force=20)
        for _ in range(120):
            p.stepSimulation()

    def move_motors(self, a, b, carrying):
        """Interpolate motor targets; stop on actual barrier contact."""
        qa, qb = self.poses[a], self.poses[b]
        # 16 target segments × 12 physics ticks = 0.8 simulated seconds.
        for segment in range(1, 17):
            f = segment / 16
            target = [u + (v - u) * f for u, v in zip(qa, qb)]
            for joint, value in enumerate(target):
                p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                        targetPosition=value, force=87, maxVelocity=1.6)
            for _ in range(12):
                p.stepSimulation()
                if p.getContactPoints(self.robot, self.obstacle):
                    return False, "robot_fixture_contact"
                if carrying and p.getContactPoints(self.parcel, self.obstacle):
                    return False, "parcel_fixture_contact"
        err = math.dist(self.tool_position(), xyz(b))
        if err > 0.025:
            return False, f"tracking_error_{err:.3f}m"
        return True, "ok"

    def image(self):
        from PIL import Image
        view = p.computeViewMatrix([1.20, -0.97, 1.34], [0.56, 0.0, 0.20], [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(58, 4 / 3, 0.05, 3.0)
        width, height, rgba, _, _ = p.getCameraImage(
            640, 480, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER)
        return Image.frombytes("RGBA", (width, height), bytes(rgba)).convert("RGB")


class SortingTask:
    def __init__(self, cell, pickup, bin_id):
        self.cell = cell
        self.pickup = pickup
        self.bin_id = bin_id
        self.node = HOME
        self.phase = 0
        self.steps = 0
        self.interlocks = 0
        self.done = False
        self.success = False

    def state(self):
        return (self.pickup, self.bin_id, self.phase, *self.node)

    def goal(self):
        return PICKS[self.pickup] if self.phase == 0 else BINS[self.bin_id]

    def step(self, action):
        if self.done:
            raise RuntimeError("episode ended")
        old = math.dist(xyz(self.node), xyz(self.goal()))
        nxt = self.cell.edges[self.phase][self.node].get(action)
        self.steps += 1
        if nxt is None:
            self.interlocks += 1
            reward, event = -0.25, "interlock"
        else:
            self.node = nxt
            reward = -0.03 + 4 * (old - math.dist(xyz(nxt), xyz(self.goal())))
            event = "move"
        if self.node == self.goal():
            if self.phase == 0:
                self.phase = 1
                reward += 3
                event = "grasp"
            else:
                self.success = self.done = True
                reward += 5
                event = "release"
        if self.steps >= MAX_STEPS and not self.done:
            self.done = True
            reward -= 3
            event = "timeout"
        return self.state(), reward, self.done, event


def choose(table, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.randrange(6)
    values = table.get(state, [0.0] * 6)
    best = max(values)
    return rng.choice([i for i, value in enumerate(values) if value >= best - 1e-10])


def train(cell, episodes, seed):
    rng = random.Random(seed)
    table, history = {}, []
    for ep in range(episodes):
        task = SortingTask(cell, rng.randrange(len(PICKS)), rng.randrange(len(BINS)))
        epsilon = max(0.03, 0.60 * (1 - ep / episodes) ** 1.5)
        for _ in range(MAX_STEPS):
            state = task.state()
            action = choose(table, state, rng, epsilon)
            next_state, reward, done, _ = task.step(action)
            values = table.setdefault(state, [0.0] * 6)
            future = 0 if done else max(table.get(next_state, [0.0] * 6))
            values[action] += 0.28 * (reward + 0.98 * future - values[action])
            if done:
                break
        history.append(int(task.success))
    return table, history


def plan(cell, table, pickup, bin_id, seed):
    task = SortingTask(cell, pickup, bin_id)
    rng = random.Random(seed)
    path = [(HOME, 0, "start")]
    for _ in range(MAX_STEPS):
        action = choose(table, task.state(), rng, 0)
        _, _, done, event = task.step(action)
        path.append((task.node, task.phase, event))
        if done:
            break
    return task, path


def replay(cell, path, pickup, bin_id, image_path=None,
           parcel_offset=(0.0, 0.0), parcel_mass=0.08, parcel_friction=0.8):
    cell.reset_joints(cell.poses[HOME])
    cell.new_parcel(pickup, parcel_offset, parcel_mass, parcel_friction)
    for _ in range(60):
        p.stepSimulation()
    images = [cell.image()] if image_path else []
    failure = None
    contact_events = 0
    tracking_errors = []
    for (old, phase_before, _), (node, _, event) in zip(path, path[1:]):
        if old != node:
            ok, message = cell.move_motors(old, node, phase_before)
            tracking_errors.append(math.dist(cell.tool_position(), xyz(node)))
            if not ok:
                failure = message
                contact_events += int("contact" in message)
                break
        if event == "grasp" and not cell.attach():
            failure = "grasp_out_of_tolerance"
            break
        if event == "release":
            cell.release()
        if image_path:
            images.append(cell.image())
    parcel_pos = p.getBasePositionAndOrientation(cell.parcel)[0]
    goal = xyz(BINS[bin_id])
    placed = (failure is None and path[-1][2] == "release"
              and math.dist(parcel_pos[:2], goal[:2]) < 0.025
              and abs(parcel_pos[2] - 0.035) < 0.03)
    if image_path and images:
        images[0].save(image_path, save_all=True, append_images=images[1:],
                       duration=180, loop=0, optimize=True)
    return {"physical_success": placed, "failure": failure,
            "parcel_final_xyz_m": [round(v, 4) for v in parcel_pos],
            "robot_fixture_contact_events": contact_events,
            "max_tracking_error_m": round(max(tracking_errors, default=0.0), 4)}


def write_report(output, report, history, gifs):
    points = []
    total = 0
    for i, success in enumerate(history):
        total += success
        if i >= 100:
            total -= history[i - 100]
        if i == 0 or i == len(history) - 1 or i % max(1, len(history) // 200) == 0:
            x = 45 + i / max(1, len(history) - 1) * 710
            y = 160 - total / min(100, i + 1) * 125
            points.append(f"{x:.1f},{y:.1f}")
    rows = []
    for pickup in range(1, len(PICKS) + 1):
        cells = []
        for bin_id in range(1, len(BINS) + 1):
            index = (pickup - 1) * len(BINS) + bin_id - 1
            outcome = "✓" if report["cases"][index]["physical_success"] else "×"
            cells.append(f'<td><button type="button" data-index="{index}" '
                         f'aria-label="取件位 {pickup} 到料箱 {bin_id}">{outcome} P{pickup}→B{bin_id}</button></td>')
        rows.append(f'<tr><th scope="row">取件位 {pickup}</th>{"".join(cells)}</tr>')
    headers = "".join(f'<th scope="col">料箱 {i}</th>'
                      for i in range(1, len(BINS) + 1))
    station_marks = []
    for name, nodes, y, colour in (("P", PICKS, 48, "#2876a7"),
                                   ("B", BINS, 142, "#2d9564")):
        for index, node in enumerate(nodes, 1):
            x = 110 + (xyz(node)[0] - X[0]) / (X[-1] - X[0]) * 570
            station_marks.append(f'<circle cx="{x:.1f}" cy="{y}" r="21" fill="{colour}"/>'
                                 f'<text x="{x:.1f}" y="{y+5}" text-anchor="middle" '
                                 f'fill="white" font-size="14">{name}{index}</text>')
    cases_json = json.dumps(report["cases"], ensure_ascii=False)
    first_image = ('<img id="selected-image" src="order_1_to_1.gif" '
                   'alt="取件位 1 到料箱 1 的三维机械臂回放">' if gifs
                   else '<p>使用 --no-gif 跳过了动画渲染</p>')
    success_count = sum(case["physical_success"] for case in report["cases"])
    stress_success = sum(c["successes"] for c in report["stress_test"]["cases"])
    stress_total = sum(c["trials"] for c in report["stress_test"]["cases"])
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>PyBullet 三维机械臂分拣</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#eef3f7;color:#183147;margin:0;padding:24px}}
    main{{max-width:1100px;margin:auto}}h1{{margin-bottom:4px}}p{{line-height:1.6}}.muted{{color:#5a6e7d}}
    .panel{{background:#fff;border-radius:14px;padding:17px;box-shadow:0 5px 22px #24415a0c;margin:16px 0}}
    .grid{{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:16px}}
    img{{width:100%;height:auto;border-radius:8px}}svg{{width:100%;height:auto;background:#f8fbfc;border-radius:8px}}
    .matrix{{width:100%;border-collapse:separate;border-spacing:5px;text-align:center}}.matrix th{{font-size:13px}}
    .matrix button{{width:100%;padding:11px 2px;border:1px solid #c7d8e3;border-radius:8px;background:#f0f7f8;color:#1a526c;cursor:pointer;font:inherit}}
    .matrix button[aria-pressed="true"]{{background:#1477a8;color:#fff;border-color:#1477a8}}
    .matrix button:hover{{border-color:#1477a8}}#selected-meta{{line-height:1.9}}
    @media(max-width:760px){{.grid{{grid-template-columns:1fr}}body{{padding:12px}}}}</style>
    <main><h1>PyBullet 三维机械臂分拣</h1><p class="muted">Franka Panda 七关节 URDF · CPU 物理仿真 · Q-learning 安全路径 · 真空吸附工具模型</p>
    <section class="panel"><h2>四个取件位 × 四个料箱</h2><svg viewBox="0 0 800 190" role="img" aria-label="四个取件位、障碍物与四个料箱的工位布局">
    <rect x="70" y="27" width="660" height="43" rx="8" fill="#d4e8f4"/>
    <rect x="70" y="120" width="660" height="43" rx="8" fill="#d9f0e2"/>
    <rect x="70" y="82" width="660" height="25" rx="4" fill="#e65b56"/>
    <text x="400" y="100" text-anchor="middle" fill="white" font-size="14">固定设备禁入区</text>
    {''.join(station_marks)}</svg><p class="muted">每个订单从一个取件位出发，越过固定设备后进入指定料箱；颜色区分独立的料箱区域。</p></section>
    <section class="panel"><h2>训练成功率</h2><svg viewBox="0 0 800 190" role="img" aria-label="最近一百回合训练成功率">
    <path d="M45 35H755 M45 160H755" stroke="#d2dfe7"/><text x="8" y="40">100%</text><text x="15" y="165">0%</text>
    <polyline points="{' '.join(points)}" fill="none" stroke="#177db5" stroke-width="3"/>
    </svg><p class="muted">默认种子 {report["seed"]} · {report["episodes"]} 回合 · 最近 100 回合 {report["last_100_train_success"]:.0%}。16 种订单的策略使用同一张 Q 表。</p></section>
    <div class="grid"><section class="panel"><h2 id="selected-title">取件位 1 → 料箱 1</h2>
    {first_image}<p id="selected-meta"></p></section>
    <section class="panel"><h2>选择订单</h2><table class="matrix"><thead><tr><th></th>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table><p class="muted">✓ 表示该订单在物理回放中完成。当前结果 {success_count}/{len(report["cases"])} 种订单完成；扰动回放 {stress_success}/{stress_total} 次完成。</p></section></div>
    <p class="muted">位姿和安全动作图来自 PyBullet 逆运动学与碰撞查询；回放使用电机控制和物理步进。包裹通过真空吸附约束保持在末端。扰动范围：包裹位置 ±5 mm、质量 0.04–0.12 kg、摩擦系数 0.4–1.0。参数是教学设定，实体迁移需标定、驱动接口与独立安全验证。</p>
    <script>const cases={cases_json};const gifs={str(gifs).lower()};
    function show(index){{const c=cases[index];document.getElementById('selected-title').textContent=`取件位 ${{c.pickup}} → 料箱 ${{c.bin}} · ${{c.physical_success?'完成':'失败'}}`;
    document.getElementById('selected-meta').textContent=`RL 路径 ${{c.steps}} 步 / 最短安全路径 ${{c.ideal_safe_steps}} 步 · 安全联锁 ${{c.interlocks}} 次 · 实际接触 ${{c.robot_fixture_contact_events}} 次 · 最大末端跟踪误差 ${{c.max_tracking_error_m}} m`;
    if(gifs){{let img=document.getElementById('selected-image');img.src=`order_${{c.pickup}}_to_${{c.bin}}.gif`;img.alt=`取件位 ${{c.pickup}} 到料箱 ${{c.bin}} 的三维机械臂回放`;}}
    document.querySelectorAll('[data-index]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.index==index?'true':'false'));}}
    document.querySelectorAll('[data-index]').forEach(b=>b.addEventListener('click',()=>show(Number(b.dataset.index))));show(0);</script>
    </main></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="also open the PyBullet 3D window")
    parser.add_argument("--no-gif", action="store_true", help="skip CPU render")
    parser.add_argument("--stress-trials", type=int, default=3,
                        help="extra physical replays per order with perturbed parcel pose/mass/friction")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if args.episodes < 1 or args.stress_trials < 0:
        parser.error("episodes must be positive and stress trials nonnegative")
    cell = BulletCell(gui=args.gui)
    try:
        cell.build_graph()
        print(f"PyBullet safety graph: {len(cell.poses)} poses, "
              f"{sum(map(len, cell.edges[0].values()))} empty-arm moves, "
              f"{sum(map(len, cell.edges[1].values()))} carrying moves")
        table, history = train(cell, args.episodes, args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        results = []
        for pickup in range(len(PICKS)):
            for bin_id in range(len(BINS)):
                task, path = plan(cell, table, pickup, bin_id,
                                  args.seed + pickup * len(BINS) + bin_id)
                gif = args.output / f"order_{pickup+1}_to_{bin_id+1}.gif"
                physical = replay(cell, path, pickup, bin_id,
                                  None if args.no_gif else gif)
                row = {"pickup": pickup + 1, "bin": bin_id + 1,
                       "policy_success": task.success, "steps": task.steps,
                       "ideal_safe_steps": cell.shortest(HOME, PICKS[pickup], 0)
                           + cell.shortest(PICKS[pickup], BINS[bin_id], 1),
                       "interlocks": task.interlocks, **physical}
                results.append(row)
                print(f"order {pickup+1}->{bin_id+1}: {row}")
        report = {"seed": args.seed, "episodes": args.episodes,
                  "last_100_train_success": sum(history[-100:]) / min(100, len(history)),
                  "safe_poses": len(cell.poses),
                  "pickup_positions_xyz_m": [xyz(node) for node in PICKS],
                  "bin_positions_xyz_m": [xyz(node) for node in BINS],
                  "cases": results}
        stress_rng = random.Random(args.seed + 991)
        stress = []
        for pickup in range(len(PICKS)):
            for bin_id in range(len(BINS)):
                _, path = plan(cell, table, pickup, bin_id,
                               args.seed + pickup * len(BINS) + bin_id)
                trials = []
                for _ in range(args.stress_trials):
                    trials.append(replay(
                        cell, path, pickup, bin_id,
                        parcel_offset=(stress_rng.uniform(-0.005, 0.005),
                                       stress_rng.uniform(-0.005, 0.005)),
                        parcel_mass=stress_rng.uniform(0.04, 0.12),
                        parcel_friction=stress_rng.uniform(0.4, 1.0)))
                stress.append({"pickup": pickup + 1, "bin": bin_id + 1,
                               "trials": len(trials),
                               "successes": sum(row["physical_success"] for row in trials),
                               "failures": [row["failure"] for row in trials
                                            if not row["physical_success"]]})
        report["stress_test"] = {"parcel_xy_jitter_m": 0.005,
                                 "parcel_mass_kg": [0.04, 0.12],
                                 "parcel_friction": [0.4, 1.0],
                                 "cases": stress}
        (args.output / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "train_success.json").write_text(
            json.dumps(history), encoding="utf-8")
        trajectories = []
        for pickup in range(len(PICKS)):
            for bin_id in range(len(BINS)):
                _, path = plan(cell, table, pickup, bin_id,
                               args.seed + pickup * len(BINS) + bin_id)
                trajectories.append({"pickup": pickup + 1, "bin": bin_id + 1,
                                     "waypoints": [
                                         {"xyz_m": xyz(node),
                                          "joint_target_rad": cell.poses[node],
                                          "tool_event": event}
                                         for node, _, event in path]})
        (args.output / "joint_trajectories.json").write_text(
            json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(args.output, report, history, not args.no_gif)
        print(f"Saved to {args.output}")
    finally:
        cell.close()


if __name__ == "__main__":
    main()
