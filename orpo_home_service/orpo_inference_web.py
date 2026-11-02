"""本地家政问答推理页：比较基础模型与 ORPO LoRA 适配器的回复。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from orpo_home_service import MODEL_ID, chat_prompt, ensure_single_openmp


ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "orpo_inference_web.html"
EFFECT_PAGE = ROOT / "orpo_effect_demo.html"
DEFAULT_ADAPTER = ROOT / "orpo_output" / "full" / "adapter"
MAX_REQUEST_BYTES = 8_192
MAX_QUESTION_CHARS = 500


class InputError(ValueError):
    """用户输入无法安全地作为一次完整推理处理。"""


class InferenceService:
    def __init__(self, model_name: str, adapter_path: Path, threads: int,
                 max_prompt: int, max_answer: int, local_files_only: bool):
        ensure_single_openmp()
        os.environ.setdefault("USE_TF", "0")
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"找不到 LoRA 适配器：{adapter_path}")
        torch.set_num_threads(threads)
        model_path = model_name
        if local_files_only and not Path(model_name).exists():
            model_path = snapshot_download(repo_id=model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, local_files_only=local_files_only)
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        self.model.eval()
        self.torch = torch
        self.max_prompt = max_prompt
        self.max_answer = max_answer
        self.lock = Lock()

    def _generate(self, inputs) -> dict:
        start = time.perf_counter()
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_answer, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        reply_ids = generated[0, inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        return {"reply": reply or "（模型未生成可见文本）",
                "seconds": round(time.perf_counter() - start, 1)}

    def compare(self, question: str) -> dict:
        # 与训练和评估使用同一个 system 提示；输入过长时拒绝而不截掉尾部问题。
        prompt = chat_prompt(self.tokenizer, question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_prompt:
            raise InputError(f"问题超过 {self.max_prompt} 个提示 token 的上限，请缩短后重试")
        with self.lock, self.torch.inference_mode():
            with self.model.disable_adapter():
                baseline = self._generate(inputs)
            tuned = self._generate(inputs)
        return {"baseline": baseline, "tuned": tuned}


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
            pages = {
                "/": (PAGE, "text/html; charset=utf-8"),
                "/orpo_effect_demo.html": (EFFECT_PAGE, "text/html; charset=utf-8"),
                "/README.md": (ROOT / "README.md", "text/markdown; charset=utf-8"),
                "/orpo_output/full/metrics.json": (ROOT / "orpo_output/full/metrics.json", "application/json; charset=utf-8"),
                "/orpo_output/full/evaluation_48.json": (ROOT / "orpo_output/full/evaluation_48.json", "application/json; charset=utf-8"),
            }
            if self.path not in pages:
                self.send_error(404)
                return
            path, content_type = pages[self.path]
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/api/compare":
                self.send_error(404)
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self.send_json(415, {"error": "请使用 JSON 格式提交问题"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.send_json(413, {"error": "请求内容过大或为空"})
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                question = payload.get("question") if isinstance(payload, dict) else None
                if not isinstance(question, str) or not question.strip():
                    self.send_json(400, {"error": "请输入家政问题"})
                    return
                question = question.strip()
                if len(question) > MAX_QUESTION_CHARS:
                    self.send_json(400, {"error": f"问题最多 {MAX_QUESTION_CHARS} 字"})
                    return
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {"error": "请求格式无效"})
                return
            try:
                self.send_json(200, service.compare(question))
            except InputError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                logging.exception("推理失败")
                self.send_json(500, {"error": "推理失败，请查看服务端日志"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face ID 或本地模型路径")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--local-files-only", action="store_true", help="仅使用已缓存的基础模型")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--max-answer", type=int, default=96)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.port <= 65535 or args.max_prompt < 32 or args.max_answer < 2:
        parser.error("threads 必须大于 0，port 必须在 1～65535，max-prompt >= 32，max-answer >= 2")
    service = InferenceService(args.model, args.adapter, args.threads,
                               args.max_prompt, args.max_answer, args.local_files_only)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"家政助手推理页面：http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
