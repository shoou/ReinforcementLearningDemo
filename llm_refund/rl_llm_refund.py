"""用单步上下文在 CPU 上微调 0.5B 语言模型完成售后分流。

所有工单均为合成数据。每张工单是一个独立的状态，模型只在 A/B/C/D
四个动作中选择；动作没有后续状态，因此这里不计算长期回报。
程序负责把动作代码映射成业务决定，模型不生成自由格式的处理意见。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


# 默认模型可由 --model 替换；动作代码必须各自对应 tokenizer 的单个 token。
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ACTIONS = ("A", "B", "C", "D")
ACTION_NAMES = {"A": "退款", "B": "补发", "C": "拒绝", "D": "人工复核"}
# 商品及描述只用于改写工单表面文本；正确动作由下面的结构化事实决定。
PRODUCTS = (
    ("一只保温杯", "杯盖拧紧后仍会漏水", "杯身和包装都完好"),
    ("一台电热水壶", "通电后指示灯不亮", "机器能正常烧水，外观也没有问题"),
    ("一副蓝牙耳机", "右耳一直没有声音", "耳机没有故障，包装也完整"),
    ("一盏台灯", "灯臂连接处断开了", "灯可以正常使用，也没有磕碰"),
    ("一把折叠伞", "伞骨有一根弯了，撑不开", "伞面和伞骨都完好"),
    ("一个收纳盒", "盒盖到货时已经裂开", "盒子完好，还没有使用"),
    ("一件外套", "袖口的缝线开了", "衣服没有破损，吊牌还在"),
    ("一个双肩包", "肩带的扣具断了", "包没有损坏，标签还在"),
)
# 与 oracle 的判断顺序一致：证据和物流问题优先于时效及商品状态。
RULES = (
    "规则：资料缺失或物流状态矛盾→D；签收超过30天→C；"
    "30天内商品破损→按客户意愿选A退款或B补发；"
    "商品完好且签收不超过7天→A；其余→C。"
)


def oracle(case: dict) -> str:
    """按显式业务规则给工单打标签，也作为奖励和评估的标准答案。

    proof 表示核验资料齐全，conflict 表示物流记录与客户描述矛盾。
    先处理需要人工复核的异常，避免后面的自动退款等规则覆盖它们。
    """
    if not case["proof"] or case["conflict"]:
        return "D"
    if case["days"] > 30:
        return "C"
    if case["damaged"]:
        return "A" if case["preference"] == "退款" else "B"
    return "A" if case["days"] <= 7 else "C"


def render_ticket(case: dict, rng: random.Random, split: str) -> str:
    """把同一组事实写成客户留言与客服备注，返回一条自然语言工单。"""
    product, defect, intact = rng.choice(PRODUCTS)
    condition = defect if case["damaged"] else intact
    if case["damaged"]:
        wish = rng.choice(("想退货退款", "希望把钱退回来", "麻烦安排退款")) if case["preference"] == "退款" else rng.choice(
            ("想换一个新的", "能否补发同款", "希望直接补发一件"))
        customer = f"{condition}，{wish}。"
    else:
        reason = rng.choice(("尺寸不太合适", "买重了", "暂时用不上", "和家里现有的款式不搭"))
        customer = f"{condition}，但{reason}，想退货退款。"
    delivery = rng.choice((f"{case['days']}天前收到货", f"收到货已经{case['days']}天", f"签收至今{case['days']}天"))
    proof = rng.choice(("订单截图和商品照片都已上传", "已补齐订单记录及商品照片",
                        "客服已核对购买记录和商品照片")) if case["proof"] else rng.choice(
                            ("目前还没有上传订单截图或商品照片", "客服尚未收到核验所需的订单记录和照片",
                             "只留了文字描述，订单凭证和照片待补"))
    logistics = rng.choice(("物流页面仍显示运输中，与客户已收货的说法不符",
                            "快递轨迹显示尚未签收，和客户描述对不上")) if case["conflict"] else rng.choice(
                                ("物流签收记录与客户描述一致", "快递轨迹和收货时间已核对",
                                 "平台显示的签收信息与客户说法相符"))
    # 各数据集使用不同句式模板，使验证和测试文本不直接复用训练模板。
    # split 只改变措辞，不改变 case 所代表的业务事实或标签。
    layouts = {
        "train": (
            f"客户留言：\"买的{product}，{delivery}。{customer}\" 客服备注：{proof}；{logistics}。",
            f"售后工单｜商品：{product}。顾客称{delivery}，{customer}核验记录：{logistics}；{proof}。",
            f"顾客申请处理{product}：{delivery}，{customer}客服查看后记录，{proof}，{logistics}。",
        ),
        "validation": (
            f"服务台记录：顾客购买{product}，{delivery}。反馈{customer}附件情况：{proof}。配送核对：{logistics}。",
            f"工单摘要：{product}，{delivery}；客户说\"{customer}\" 平台核验到{proof}，{logistics}。",
        ),
        "test": (
            f"线上售后留言：\"{product}{delivery}。{customer}\" 处理前核查：{logistics}；{proof}。",
            f"客服转来的订单：买家说{product}{delivery}，{customer}系统备注{proof}，{logistics}。",
        ),
    }
    return rng.choice(layouts[split])


def make_cases(seed: int, count: int, split: str) -> list[dict]:
    """生成四类标签近似均衡、文本不重复的合成工单。

    先按目标标签抽样满足规则的事实，再用 oracle 校验。不同数据集使用
    不同随机种子和文本模板；相同种子则可复现相同数据。
    """
    rng = random.Random(seed)
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    cases = []
    seen_text = set()
    for i in range(count):
        # 轮流指定 A/B/C/D，避免单一类别占据训练或评估数据的大部分。
        label = ACTIONS[i % 4]
        if label == "A":
            damaged = rng.choice([True, False])
            days = rng.choice((1, 2, 6, 7, 8, 29, 30)) if damaged else rng.randint(1, 7)
            preference = "退款"
        elif label == "B":
            damaged, days, preference = True, rng.choice((1, 3, 7, 8, 15, 29, 30)), "补发"
        elif label == "C":
            damaged = rng.choice([True, False])
            days = rng.choice((31, 32, 40, 50)) if damaged or rng.random() < 0.5 else rng.choice((8, 9, 15, 29, 30))
            preference = rng.choice(["退款", "补发"]) if damaged else "退款"
        else:
            damaged, days = rng.choice([True, False]), rng.randint(1, 50)
            preference = rng.choice(["退款", "补发"]) if damaged else "退款"
        # D 至少满足“缺资料”或“物流矛盾”之一；其他标签两项均正常。
        proof = label != "D" or rng.random() < 0.5
        conflict = label == "D" and (proof or rng.random() < 0.5)
        case = dict(days=days, damaged=damaged, preference=preference,
                    proof=proof, conflict=conflict)
        # 仅在当前 split 内去重；遇到碰撞时重新抽样文本措辞。
        for _ in range(100):
            case["text"] = render_ticket(case, rng, split)
            if case["text"] not in seen_text:
                break
        else:
            raise RuntimeError("Could not generate enough distinct ticket texts")
        seen_text.add(case["text"])
        # 生成器的目标标签与业务规则必须一致，否则奖励会自相矛盾。
        assert oracle(case) == label
        cases.append(case)
    rng.shuffle(cases)
    return cases


def export_cases(directory: str, seed: int, train_size: int, eval_size: int) -> None:
    """导出可检查的 JSONL 数据；此路径不加载模型或训练依赖。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for split, offset, size in (("train", 0, train_size),
                                ("validation", 1, eval_size), ("test", 2, eval_size)):
        cases = make_cases(seed + offset, size, split)
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                record = {"text": case["text"], "action": oracle(case),
                          "action_name": ACTION_NAMES[oracle(case)],
                          "facts": {key: case[key] for key in ("days", "damaged", "preference", "proof", "conflict")}}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(cases)} -> {target / (split + '.jsonl')}")


def reward(case: dict, action: str) -> float:
    """计算单步奖励：正确为 1，普通错误为 -1，漏掉人工复核为 -2。"""
    correct = oracle(case)
    if action == correct:
        return 1.0
    if correct == "D" and action != "D":
        return -2.0  # 证据待核验时自动决策的风险更高，给予更大惩罚。
    return -1.0


def prompt(case: dict, tokenizer) -> str:
    """把规则和工单填入模型的对话模板，并定位到助手开始回答处。"""
    messages = [
        {"role": "system", "content": "你是售后工单分流员。只选一个动作代码，不要解释。"},
        {"role": "user", "content": RULES + "\n" + case["text"] + "\nA退款 B补发 C拒绝 D人工复核。只输出代码："},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def action_token_ids(tokenizer) -> list[int]:
    """取得四个动作代码的 token ID；多 token 编码无法直接作四分类。"""
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ACTIONS]
    if any(len(x) != 1 for x in ids):
        raise ValueError("This tokenizer does not encode A/B/C/D as single tokens")
    return [x[0] for x in ids]


def action_logits(model, tokenizer, cases: list[dict], token_ids: list[int]):
    """取每条提示末尾预测下一 token 的分数，并只保留 A/B/C/D。"""
    import torch
    batch = tokenizer([prompt(c, tokenizer) for c in cases], return_tensors="pt",
                      padding=True, truncation=True, max_length=384)
    # 批次右侧补齐后，各行最后一个真实 token 的位置不同；不能直接取 logits[:, -1]。
    output = model(**batch, use_cache=False).logits
    ends = batch["attention_mask"].sum(dim=1) - 1
    return output[torch.arange(len(cases)), ends][:, token_ids].float()


def evaluate(model, tokenizer, cases: list[dict], token_ids: list[int], batch_size: int) -> dict:
    """以四选一贪心决策评估准确率、平均奖励和危险自动决策数。"""
    import torch
    model.eval()
    correct = risky = total_reward = 0
    # 混淆矩阵第一层是真实动作，第二层是模型选择的动作。
    confusion = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            choices = action_logits(model, tokenizer, batch, token_ids).argmax(dim=-1).tolist()
            for case, index in zip(batch, choices):
                truth, chosen = oracle(case), ACTIONS[index]
                confusion[truth][chosen] += 1
                correct += truth == chosen
                risky += truth == "D" and chosen != "D"
                total_reward += reward(case, chosen)
    return {"accuracy": correct / len(cases), "mean_reward": total_reward / len(cases),
            "risky_auto_decisions": risky, "n": len(cases), "confusion": confusion}


def ensure_single_openmp() -> None:
    """在 macOS Conda 环境中，令 OpenBLAS 复用 pip PyTorch 附带的 libomp。

    只有检测到两份不同的 libomp 且搜索路径尚未包含 PyTorch 版本时才
    重启当前进程；环境标记用于防止重复重启。
    """
    if sys.platform != "darwin" or os.environ.get("RL_LLM_OPENMP_REEXEC"):
        return
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.submodule_search_locations:
        return
    torch_lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    torch_omp = torch_lib / "libomp.dylib"
    conda_omp = Path(sys.prefix) / "lib" / "libomp.dylib"
    if not (torch_omp.exists() and conda_omp.exists()) or torch_omp.samefile(conda_omp):
        return
    paths = os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
    if str(torch_lib) in paths:
        return
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib)] + [p for p in paths if p])
    env["RL_LLM_OPENMP_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def run(args) -> None:
    """加载模型、按需训练 LoRA，并保存最佳适配器及评估报告。"""
    ensure_single_openmp()
    # 本程序只用 PyTorch；避免 transformers 探测并导入环境中的 TensorFlow。
    # 旧版 TensorFlow 的二进制扩展可能与 NumPy 2 不兼容。
    os.environ["USE_TF"] = "0"
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 因果语言模型通常没有独立 pad token；补齐只用于批量前向计算。
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_ids = action_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    # 已有适配器只用于评估；新训练仅更新 LoRA 注入的少量参数。
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    elif args.steps:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))

    train = make_cases(args.seed, args.train_size, "train")
    valid = make_cases(args.seed + 1, args.eval_size, "validation")
    test = make_cases(args.seed + 2, args.eval_size, "test") if not args.validation_only else None
    # 训练前基线在验证集上测量；加载已有适配器时不另算基础模型基线。
    baseline = None if args.adapter else evaluate(model, tokenizer, valid, token_ids, args.batch_size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_valid = None
    if args.steps and not args.adapter:
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        for step in range(1, args.steps + 1):
            # 按固定顺序循环取样，跨过训练集末尾时从头继续。
            batch = [train[((step - 1) * args.batch_size + i) % len(train)] for i in range(args.batch_size)]
            logits = action_logits(model, tokenizer, batch, token_ids)
            # 只在四个合法动作上归一化，得到当前策略 pi(a | 工单)。
            log_probs = torch.log_softmax(logits, dim=-1)
            # 策略网络输出的动作概率分布（例如，当前 Token 的词表概率分布）。
            probs = log_probs.exp()
            # 环境或奖励模型（Reward Model）给出的每个动作的分数，业务规则给出四个动作各自的奖励。
            rewards = torch.tensor([[reward(c, a) for a in ACTIONS] for c in batch])
            # 最大化 E_{a~pi}[r(s,a)]：逐项相乘后求和，是精确的单步期望奖励。
            expected_reward = (probs * rewards).sum(dim=-1).mean()
            # 熵项鼓励策略保留一定分散度；梯度只通过 probs 回传到 LoRA 参数。
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = -expected_reward - args.entropy * entropy
            optimizer.zero_grad()
            loss.backward()
            # 限制梯度范数，减少偶发的大幅参数更新。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, tokenizer, valid, token_ids, args.batch_size)
                row = {"step": step, "loss": float(loss.detach()), "validation": metrics}
                history.append(row)
                print(json.dumps(row, ensure_ascii=False))
                # 先比较验证集平均奖励，再用准确率打破平局。
                if best_valid is None or (metrics["mean_reward"], metrics["accuracy"]) > best_valid:
                    best_valid = (metrics["mean_reward"], metrics["accuracy"])
                    model.save_pretrained(output / "adapter")
                model.train()
        # 重新加载验证集选中的最佳权重，再对测试集作最终评估。
        # --validation-only 时 test 为 None，不会读取或评估测试集。
        del model
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32),
            output / "adapter", is_trainable=False)
    final_valid = evaluate(model, tokenizer, valid, token_ids, args.batch_size)
    final_test = evaluate(model, tokenizer, test, token_ids, args.batch_size) if test else None
    report = {"model": args.model, "config": vars(args),
              "baseline_validation": baseline, "final_validation": final_valid,
              "final_test": final_test, "history": history}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_validation": baseline, "final_validation": final_valid,
                      "final_test": final_test}, ensure_ascii=False, indent=2))
    for case in (test or valid)[:4]:
        choice = ACTIONS[action_logits(model, tokenizer, [case], token_ids).argmax(dim=-1).item()]
        print(json.dumps({"ticket": case["text"], "decision": ACTION_NAMES[choice],
                          "expected": ACTION_NAMES[oracle(case)]}, ensure_ascii=False))


def main() -> None:
    """解析命令行；导出数据与加载模型训练是两条独立执行路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face ID or local model directory")
    parser.add_argument("--export-data", help="Write train/validation/test JSONL and exit; no model download")
    parser.add_argument("--adapter", help="Existing LoRA adapter; evaluates it without training")
    parser.add_argument("--validation-only", action="store_true", help="Tune without opening the test split")
    parser.add_argument("--output", default="rl_llm_output")
    parser.add_argument("--steps", type=int, default=128, help="0 runs a baseline evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.train_size < 1 or args.eval_size < 4 or args.steps < 0 or args.eval_every < 1:
        parser.error("batch/train sizes and eval-every must be positive; eval-size >= 4; steps >= 0")
    if args.export_data:
        export_cases(args.export_data, args.seed, args.train_size, args.eval_size)
        return
    run(args)


if __name__ == "__main__":
    main()
