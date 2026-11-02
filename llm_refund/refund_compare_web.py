"""Local baseline/LoRA comparison page for rl_llm_refund.py."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from rl_llm_refund import (
    ACTION_NAMES,
    ACTIONS,
    MODEL_ID,
    action_logits,
    action_token_ids,
    ensure_single_openmp,
)

PAGE = Path(__file__).with_name("refund_compare_web.html")
MAX_REQUEST_BYTES = 16_384
MAX_TICKET_CHARS = 2_000


class ComparisonModel:
    def __init__(self, model_name: str, adapter_path: str, threads: int):
        ensure_single_openmp()
        # 只使用 PyTorch，避免 transformers 导入与 NumPy 2 不兼容的 TensorFlow。
        os.environ["USE_TF"] = "0"
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = Path(adapter_path)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"找不到 LoRA 适配器：{adapter}")
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.token_ids = action_token_ids(self.tokenizer)
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.eval()
        self.lock = Lock()

    def compare(self, ticket: str) -> dict:
        # Use the same prompt and next-token A/B/C/D logits as the training evaluator.
        case = {"text": ticket}
        with self.lock, self.torch.inference_mode():
            with self.model.disable_adapter():
                baseline = self._score(case)
            tuned = self._score(case)
        return {"baseline": baseline, "tuned": tuned, "changed": baseline["action"] != tuned["action"]}

    def _score(self, case: dict) -> dict:
        logits = action_logits(self.model, self.tokenizer, [case], self.token_ids)[0]
        probabilities = self.torch.softmax(logits, dim=-1).tolist()
        chosen = ACTIONS[max(range(len(ACTIONS)), key=lambda i: probabilities[i])]
        return {
            "action": chosen,
            "action_name": ACTION_NAMES[chosen],
            "scores": {action: round(float(probabilities[i]), 6) for i, action in enumerate(ACTIONS)},
        }


def make_handler(service):
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
            if self.path != "/":
                self.send_error(404)
                return
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/api/compare":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.send_json(413, {"error": "请求内容过大或为空"})
                    return
                payload = json.loads(self.rfile.read(length))
                ticket = payload.get("ticket")
                if not isinstance(ticket, str) or not ticket.strip():
                    self.send_json(400, {"error": "请输入售后问题"})
                    return
                ticket = ticket.strip()
                if len(ticket) > MAX_TICKET_CHARS:
                    self.send_json(400, {"error": f"问题最多 {MAX_TICKET_CHARS} 字"})
                    return
            except (ValueError, TypeError, UnicodeDecodeError, AttributeError):
                self.send_json(400, {"error": "请求格式无效"})
                return
            try:
                self.send_json(200, service.compare(ticket))
            except Exception as exc:
                self.send_json(500, {"error": f"推理失败：{exc}"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face ID 或本地模型路径")
    parser.add_argument("--adapter", default="rl_llm_output/run_01/adapter")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("threads 必须大于 0，port 必须在 1 到 65535 之间")
    service = ComparisonModel(args.model, args.adapter, args.threads)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"售后决策对比页面：http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
