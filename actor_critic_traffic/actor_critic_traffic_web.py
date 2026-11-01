"""用本地网页逐分钟展示训练好的信号灯策略与固定配时。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch

from actor_critic_traffic import ActorCritic, HORIZON, TrafficLight, fixed_timing


PAGE = Path(__file__).with_name("actor_critic_traffic_web.html")
DEFAULT_MODEL = Path("actor_critic_output/actor_critic_model.pt")


class Simulation:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到模型文件：{model_path}；请先运行 actor_critic_traffic.py 训练模型")
        self.model = ActorCritic()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def run(self, seed: int) -> dict:
        env = TrafficLight(seed)
        state = env.reset()
        learned_records = []
        with torch.inference_mode():
            for minute in range(1, HORIZON + 1):
                logits, value = self.model(torch.from_numpy(state).unsqueeze(0))
                probabilities = torch.softmax(logits, dim=-1)[0].tolist()
                action = int(logits.argmax(dim=-1).item())
                state, reward, _, info = env.step(action)
                learned_records.append({
                    "minute": minute,
                    "action": action,
                    "arrivals": info["arrivals"].tolist(),
                    "queues": info["queues"].tolist(),
                    "passed": info["passed"],
                    "switched": info["switched"],
                    "waiting": info["waiting"],
                    "reward": round(reward, 3),
                    "probabilities": [round(p, 4) for p in probabilities],
                    "value": round(float(value.item()), 3),
                    "total_wait": env.total_wait,
                    "total_passed": env.total_passed,
                    "switches": env.switches,
                })

        fixed = fixed_timing(seed)
        fixed_records = []
        total_wait = total_passed = switches = 0
        for record in fixed["records"]:
            total_wait += record["waiting"]
            total_passed += record["passed"]
            switches += int(record["switched"])
            fixed_records.append({
                "minute": record["t"],
                "action": record["action"],
                "arrivals": record["arrivals"].tolist(),
                "queues": record["queues"].tolist(),
                "passed": record["passed"],
                "switched": record["switched"],
                "waiting": record["waiting"],
                "total_wait": total_wait,
                "total_passed": total_passed,
                "switches": switches,
            })
        return {"seed": seed, "horizon": HORIZON, "learned": learned_records, "fixed": fixed_records}


def make_handler(simulation: Simulation):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path != "/api/simulate":
                self.send_error(404)
                return
            try:
                raw_seed = parse_qs(parsed.query).get("seed", ["100000"])[0]
                seed = int(raw_seed)
                if not 0 <= seed <= 2**32 - 1:
                    raise ValueError
            except ValueError:
                self.send_json(400, {"error": "随机种子必须是 0 到 4294967295 之间的整数"})
                return
            self.send_json(200, simulation.run(seed))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="训练好的模型权重路径")
    parser.add_argument("--port", type=int, default=8765, help="本地网页端口")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    simulation = Simulation(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(simulation))
    print(f"路口模拟展示：http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
