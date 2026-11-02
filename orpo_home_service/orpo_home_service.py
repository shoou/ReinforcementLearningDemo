"""CPU 上运行的 ORPO 教学案例：中文家政预约助手的偏好微调。

流程：生成 chosen/rejected 偏好对 → 用 LoRA 和 ORPO 损失微调 → 在独立数据上
比较回答偏好与风险用语。所有用户请求及偏好答案都是合成样本，不会创建真实预约。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# POLICY 同时进入训练与推理时的 system 消息，定义助手可以确认的事实边界。
# 模型不会连接报价或排班服务，因此“已预约”“保证有空档”等说法属于错误偏好。
POLICY = (
    "你是社区家政预约助手。根据用户提供的信息给出简短、具体的下一步。"
    "未确认服务面积、日期和时段时先询问缺失信息；价格须以实际报价为准；"
    "未接入预约系统，不得声称已经预约或保证有空档；服务范围不明确时转人工确认。"
)
SERVICES = ("日常保洁", "深度保洁", "厨房清洁", "玻璃清洁")
TIMES = ("周六上午", "周日下午", "下周二上午", "下周五下午")
AREAS = ("45", "68", "82", "110")
CATEGORIES = ("missing_area", "missing_time", "price", "availability", "scope", "reschedule")
# 各划分使用不同问句开头，减少完全相同 prompt 跨集合出现；
# 这只防止字面重复，不等于消除类别和答案模板之间的语义相似性。
QUESTION_STYLES = {
    "train": ("您好，{body}", "想咨询一下：{body}", "家里需要安排服务，{body}"),
    "validation": ("请问{body}", "我准备找家政，{body}"),
    "test": ("麻烦帮我看看，{body}", "有个家政需求：{body}"),
}


def make_pair(category: str, service: str, area: str, when: str, rng: random.Random, split: str) -> dict:
    """按业务情形生成一对回答：chosen 遵守边界，rejected 包含不当承诺。"""
    # 六类情形分别测试补充信息、报价、排班、服务范围和改期请求。
    # 每行的 chosen/rejected 共享同一用户问题，供成对偏好损失直接比较。
    if category == "missing_area":
        body = f"{when}想做一次{service}，能帮我安排吗？"
        good = f"可以先登记需求。请问房屋大约多少平方米？有了面积后才能进一步确认{when}的安排。"
        bad = f"已经帮您预约好{when}的{service}，上门费用固定是199元。"
    elif category == "missing_time":
        body = f"我家约{area}平方米，想做{service}，什么时候能上门？"
        good = f"请问您希望哪一天、哪个时段上门？收到时间后我可以帮您整理{area}平方米的{service}需求。"
        bad = f"我们保证明天上午有空档，已经给您安排师傅上门。"
    elif category == "price":
        body = f"{area}平方米，{when}做{service}要多少钱？"
        good = f"已记录{area}平方米、{when}和{service}。费用还需结合具体服务范围核实，以实际报价为准；需要我帮您整理询价信息吗？"
        bad = f"{area}平方米的{service}固定收费199元，不会有任何变化。"
    elif category == "availability":
        body = f"{area}平方米的{service}想约{when}，这个时段还有人吗？"
        good = f"我没有实时排班信息，不能确认{when}是否有空档。我可以记录{area}平方米的{service}需求，待工作人员核实后回复您。"
        bad = f"{when}肯定有空档，已经预约成功，您直接等师傅上门即可。"
    elif category == "scope":
        body = f"{area}平方米想在{when}做{service}，还要清洗高处外墙，可以一起做吗？"
        good = f"高处外墙是否属于服务范围需要人工确认。我可以先记录{area}平方米、{when}的{service}需求，再请工作人员核实附加项目。"
        bad = "高处外墙清洗当然包含在内，不需要确认，也不用额外报价。"
    else:
        body = f"之前想约{when}给{area}平方米的家做{service}，现在想改时间，能直接改好吗？"
        good = "可以帮您记录改期请求。请告诉我新的日期和时段；具体是否可改，还需工作人员核对原预约与排班。"
        bad = "已经替您改期成功，新时间定为明天上午，不用再确认。"
    question = rng.choice(QUESTION_STYLES[split]).format(body=body)
    return {"prompt": question, "chosen": good, "rejected": bad, "category": category}


def make_dataset(seed: int, per_category: int, split: str) -> list[dict]:
    """每类生成固定数量的唯一 prompt，再打乱成一个数据划分。"""
    rng = random.Random(seed)
    rows, seen = [], set()
    for category in CATEGORIES:
        attempts = 0
        while sum(row["category"] == category for row in rows) < per_category:
            # 服务、面积、时间与问句模板的组合有限；设置尝试上限防止死循环。
            attempts += 1
            if attempts > 5000:
                raise RuntimeError("Not enough distinct synthetic prompts")
            row = make_pair(category, rng.choice(SERVICES), rng.choice(AREAS), rng.choice(TIMES), rng, split)
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            rows.append(row)
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """逐行写入 UTF-8 JSON，中文保留原文而非转义码。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """读取偏好对，并确认训练所需的四个字段均非空。"""
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not all(row.get(key) for key in ("prompt", "chosen", "rejected", "category")) for row in rows):
        raise ValueError(f"Invalid or empty preference dataset: {path}")
    return rows


def export_data(directory: Path, seed: int, train_per_category: int, eval_per_category: int) -> None:
    """生成训练、验证、测试集，并检查 prompt 字面值不跨划分重复。"""
    splits = (("train", seed, train_per_category), ("validation", seed + 1, eval_per_category),
              ("test", seed + 2, eval_per_category))
    seen = set()
    for split, split_seed, count in splits:
        rows = make_dataset(split_seed, count, split)
        # Different sentence frames keep identical prompts out of later splits.
        overlap = seen.intersection(row["prompt"] for row in rows)
        if overlap:
            raise AssertionError(f"Prompt leakage in {split}: {len(overlap)}")
        seen.update(row["prompt"] for row in rows)
        write_jsonl(directory / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} pairs -> {directory / (split + '.jsonl')}")


def ensure_single_openmp() -> None:
    """macOS Conda 同时带有两份 libomp 时，以 Torch 的库路径重新启动进程。"""
    if sys.platform != "darwin" or os.environ.get("ORPO_OPENMP_REEXEC"):
        return
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.submodule_search_locations:
        return
    torch_lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    torch_omp = torch_lib / "libomp.dylib"
    conda_omp = Path(sys.prefix) / "lib" / "libomp.dylib"
    if not torch_omp.exists() or not conda_omp.exists() or torch_omp.samefile(conda_omp):
        return
    # 只重启一次；标志位防止再次进入本分支形成递归重启。
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib), env.get("DYLD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    env["ORPO_OPENMP_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def chat_prompt(tokenizer, question: str) -> str:
    """用模型自带聊天模板组装 system+user 消息，并预留 assistant 生成前缀。"""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": POLICY}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def completion_logp(model, tokenizer, prefix: str, answer: str, max_prompt: int, max_answer: int):
    """计算回答 token 的平均对数概率；提示词仅作为条件，不计入得分。"""
    import torch
    # 截取提示词尾部，通常保留最近的用户信息及 assistant 起始标记。
    # 回答预留一个位置给 EOS，使 chosen/rejected 都含有结束信号。
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)[-max_prompt:]
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)[:max_answer - 1] + [tokenizer.eos_token_id]
    if not answer_ids:
        raise ValueError("Empty answer after tokenization")
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long)
    # 因果语言模型在位置 i 的输出预测位置 i+1：从提示词最后一个位置开始取 logits，
    # 恰好与 answer_ids（含 EOS）逐 token 对齐；提示词部分不会进入最终均值。
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1:-1, :]
    targets = ids[0, len(prompt_ids):]
    return torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).mean()


def pair_scores(model, tokenizer, row: dict, args):
    """为同一问题下的 chosen/rejected 分别求回答条件概率得分。"""
    prefix = chat_prompt(tokenizer, row["prompt"])
    return (completion_logp(model, tokenizer, prefix, row["chosen"], args.max_prompt, args.max_answer),
            completion_logp(model, tokenizer, prefix, row["rejected"], args.max_prompt, args.max_answer))


def orpo_loss(chosen_logp, rejected_logp, beta: float):
    """ORPO：chosen 的负对数似然，加 beta 倍的偏好赔率损失。"""
    import torch
    # 令 p=exp(平均 token logp)，则 log odds = log(p/(1-p))。
    # 平均 logp 理应为负；clamp 防止 p 接近 1 时 log(1-p) 数值发散。
    chosen = chosen_logp.clamp(max=-1e-7)
    rejected = rejected_logp.clamp(max=-1e-7)
    log_odds = (chosen - torch.log1p(-chosen.exp())) - (rejected - torch.log1p(-rejected.exp()))
    # softplus(-差值) 等价于 -log sigmoid(差值)：chosen 赔率越高，惩罚越小。
    # 第一项维持 chosen 回答的语言建模质量，第二项推动其相对 rejected 更可取。
    return -chosen_logp + beta * torch.nn.functional.softplus(-log_odds)


def preference_accuracy(model, tokenizer, rows: list[dict], args) -> dict:
    """计算 chosen 得分高于 rejected 的配对比例及平均 logp 差值。"""
    import torch
    model.eval()
    wins, margin = 0, 0.0
    with torch.no_grad():
        for row in rows:
            chosen, rejected = pair_scores(model, tokenizer, row, args)
            # 这里只比较给定答案的概率，并不判断模型自由生成回答是否安全。
            difference = (chosen - rejected).item()
            wins += difference > 0
            margin += difference
    return {"pairs": len(rows), "accuracy": wins / len(rows), "mean_logp_margin": margin / len(rows)}


def sample_replies(model, tokenizer, rows: list[dict], args) -> list[dict]:
    """每类抽一个问题，贪心生成示例回答并标记简单风险词。"""
    import torch
    model.eval()
    selected = [next(row for row in rows if row["category"] == category) for category in CATEGORIES]
    output = []
    with torch.no_grad():
        for row in selected:
            prompt = chat_prompt(tokenizer, row["prompt"])
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt)
            generated = model.generate(**tokens, max_new_tokens=args.max_answer, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(generated[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            # 这是可解释的关键词启发式，只检查明确金额或若干绝对承诺措辞；
            # 未命中不代表回答正确、合规或确实可执行。
            risky = bool(re.search(r"\d+(?:\.\d+)?\s*(?:元|块)(?:\s*/\s*(?:小时|次|平方米))?", reply)) or any(
                term in reply for term in ("预约成功", "已预约好", "保证有空档", "固定收费", "已经改期成功", "没有其他客户在预定"))
            output.append({"category": row["category"], "prompt": row["prompt"], "reply": reply,
                           "risk_phrase_detected": risky})
    return output


def run(args) -> None:
    """加载模型与数据，按参数训练或加载适配器，再做验证和独立测试。"""
    os.environ.setdefault("USE_TF", "0")
    ensure_single_openmp()
    # 重型模型依赖放在 run 内，--export-data 模式可在没有这些包时生成 JSONL。
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    data_dir = Path(args.data_dir)
    def balanced_subset(path: Path) -> list[dict]:
        """轮流取六类样本，使少量验证/测试对也尽量覆盖所有情形。"""
        rows = read_jsonl(path)
        by_category = {category: [row for row in rows if row["category"] == category] for category in CATEGORIES}
        if any(not group for group in by_category.values()):
            raise ValueError(f"Every category must occur in {path}")
        result = []
        while len(result) < min(args.eval_pairs, len(rows)):
            for category in CATEGORIES:
                if by_category[category] and len(result) < min(args.eval_pairs, len(rows)):
                    result.append(by_category[category].pop(0))
        return result

    valid = balanced_subset(data_dir / "validation.jsonl")
    test = balanced_subset(data_dir / "test.jsonl")
    # steps=0 时只做现有模型/适配器评估，不读取训练集。
    train = read_jsonl(data_dir / "train.jsonl") if args.steps else []
    model_path = args.model
    if args.local_files_only and not Path(args.model).exists():
        # 离线模式下将仓库 ID 解析为本机缓存快照路径，不触发模型下载。
        model_path = snapshot_download(repo_id=args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
                                                 local_files_only=args.local_files_only)
    if args.adapter:
        # 评估已有 LoRA：加载后冻结，不执行训练。
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    elif args.steps:
        # 新训练只在注意力的 q_proj/v_proj 上添加低秩参数；底座模型保持冻结。
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                               lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                               task_type=TaskType.CAUSAL_LM))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    before = preference_accuracy(model, tokenizer, valid, args) if not args.adapter else None
    before_replies = sample_replies(model, tokenizer, valid, args) if not args.adapter else []
    history = []
    if args.steps:
        # 优化器仅接收 requires_grad=True 的 LoRA 参数；每步处理一组偏好对。
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        shuffled = train[:]
        random.shuffle(shuffled)
        for step in range(1, args.steps + 1):
            model.train()
            # 数据先打乱，走完一遍后循环复用；不在每轮重新洗牌。
            row = shuffled[(step - 1) % len(shuffled)]
            chosen, rejected = pair_scores(model, tokenizer, row, args)
            loss = orpo_loss(chosen, rejected, args.beta)
            optimizer.zero_grad()
            loss.backward()
            # 限制梯度范数，降低个别长回答造成更新幅度过大的风险。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                # 验证集只用于观察训练过程，不参与梯度更新。
                metrics = preference_accuracy(model, tokenizer, valid, args)
                history.append({"step": step, "loss": float(loss.detach()), "validation": metrics})
                print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        model.save_pretrained(output / "adapter")
    # 最终测试集与训练、验证集分离；结果同时保留数值指标和每类示例回复。
    after = preference_accuracy(model, tokenizer, valid, args)
    after_replies = sample_replies(model, tokenizer, valid, args)
    final_test = preference_accuracy(model, tokenizer, test, args)
    result = {"model": args.model, "config": vars(args), "baseline_validation": before,
              "final_validation": after, "final_test": final_test, "history": history,
              "before_replies": before_replies, "after_replies": after_replies,
              "risk_phrase_count_before": sum(x["risk_phrase_detected"] for x in before_replies),
              "risk_phrase_count_after": sum(x["risk_phrase_detected"] for x in after_replies)}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("baseline_validation", "final_validation", "final_test",
                                                "risk_phrase_count_before", "risk_phrase_count_after")},
                     ensure_ascii=False, indent=2))


def main() -> None:
    """解析数据规模、训练和评估选项；导出数据与运行模型是两个入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-data", action="store_true", help="Generate JSONL data without model dependencies")
    parser.add_argument("--data-dir", default=str(Path(__file__).with_name("preference_data")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("orpo_output")))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", help="Evaluate a saved LoRA adapter; requires --steps 0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train-per-category", type=int, default=40)
    parser.add_argument("--eval-per-category", type=int, default=8)
    parser.add_argument("--eval-pairs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-prompt", type=int, default=160)
    parser.add_argument("--max-answer", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.eval_pairs < len(CATEGORIES) or min(args.train_per_category, args.eval_per_category, args.eval_every,
           args.rank, args.max_prompt, args.threads) < 1 or args.max_answer < 2 or args.steps < 0 or args.beta < 0:
        parser.error("sizes, rank, lengths and threads must be positive; eval-pairs >= 6; max-answer >= 2; steps and beta nonnegative")
    if args.adapter and args.steps:
        parser.error("--adapter requires --steps 0")
    if args.export_data:
        # 只生成合成数据，不加载模型依赖，也不执行微调。
        export_data(Path(args.data_dir), args.seed, args.train_per_category, args.eval_per_category)
    else:
        run(args)


if __name__ == "__main__":
    main()
