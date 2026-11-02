# ORPO 微调案例：社区家政预约助手

本案例让 0.5B 中文指令模型学习家政预约中的服务偏好：信息不全时询问、价格以实际报价为准、没有排班信息时不承诺空档、特殊服务转人工确认。所有数据均为合成课堂材料，脚本不会连接预约系统，也不会创建真实工单。

## 效果演示

直接打开 [效果演示页面](orpo_effect_demo.html)，可切换六类家政情形，对照基础模型与 ORPO 微调后的实际生成回复，并查看验证集偏好准确率的训练曲线。页面内嵌本案例已保存的实验快照，无需启动服务或加载模型；它展示的是既有问题的记录，不会对新输入实时推理。指标和回复分别取自 [`metrics.json`](orpo_output/full/metrics.json) 与 [`evaluation_48.json`](orpo_output/full/evaluation_48.json)。

## 实时推理页面

安装下文列出的模型依赖并下载（或缓存）基础模型后，在本目录启动本机服务：

```bash
python orpo_inference_web.py --local-files-only
```

如果基础模型尚未缓存，去掉 `--local-files-only`，首次运行会从 Hugging Face 下载。启动后打开 [http://127.0.0.1:7861/](http://127.0.0.1:7861/)；页面可输入新问题，现场并排生成基础模型与 `orpo_output/full/adapter` 微调模型的回复。六个示例按钮只是填入问题，提交后仍会重新推理。服务仅监听 `127.0.0.1`，按 `Ctrl+C` 停止；可用 `--port`、`--threads`、`--model`、`--adapter` 调整运行参数。CPU 推理可能较慢。页面和模型均不会查询真实报价、排班或创建预约。

## ORPO 学习什么

每条数据包含相同用户问题下的较好回复 `chosen` 和较差回复 `rejected`。例如：

```json
{"prompt":"周六上午想做一次厨房清洁，能帮我安排吗？","chosen":"可以先登记需求。请问房屋大约多少平方米？","rejected":"已经预约好，费用固定是199元。","category":"missing_area"}
```

ORPO 在较好回复的负对数似然（SFT）上，加一个区分两种回复的 odds ratio 损失；它不需要另载参考模型。本脚本直接用 PyTorch 实现这个目标，训练 Qwen2.5-0.5B-Instruct 的 LoRA 参数。实现使用**回复 token 的平均对数概率**计算 odds ratio，以减轻长短回复对偏好比较的影响：

```text
L = -log p(chosen) + beta * softplus(-(logit(p(chosen)) - logit(p(rejected))))
```

这里的 `log p` 是每 token 的平均对数概率，`logit(p) = log(p) - log(1-p)`。代码中只对回复 token（含结束标记）计分，不把用户问题作为训练目标。参见 [ORPO 论文](https://aclanthology.org/2024.emnlp-main.626/) 和 [TRL 的 ORPO 说明](https://huggingface.co/docs/trl/orpo_trainer)。TRL 在当前环境不可下载，因此脚本没有依赖 TRL，也不声称使用 `ORPOTrainer`。

## 数据

`preference_data/` 中有训练 240 对、验证 48 对、测试 48 对，六类需求各占相同数量：面积缺失、时段缺失、询价、询问空档、服务范围、改期。训练、验证、测试分别使用不同随机种子及提问开头，生成时检查三组问题文本无重复。回复是规则编写的合成文字，不是人工标注的真实偏好。

不加载模型即可重新生成数据：

```bash
python3 orpo_home_service.py --export-data
```

如需不同规模，可调整 `--train-per-category` 和 `--eval-per-category`。重新生成的数据会覆盖三个 JSONL 文件。

## CPU 运行

建议 Python 3.10+、至少 8 GB 可用内存；16 GB 更从容。首次运行需下载基础模型，离线运行须已缓存模型。安装 CPU 版 PyTorch、Transformers 和 PEFT 即可；本机已验证的版本为 PyTorch 2.14.1、Transformers 4.57.6、PEFT 0.21.2。新环境的安装示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch "transformers>=4.57,<5" "peft>=0.21,<1"
```

在本目录运行快速检查：

```bash
python orpo_home_service.py --steps 2 --eval-pairs 6 --max-answer 32 --output orpo_output/smoke
```

完整教学实验（240 步，每条训练样本参与一次更新）：

```bash
python orpo_home_service.py --steps 240 --eval-every 60 --eval-pairs 24 --output orpo_output/full
```

`--steps 0` 只评估基础模型。`--adapter orpo_output/full/adapter --steps 0` 可重新评估保存的 LoRA 适配器。离线且模型已缓存时可加 `--local-files-only`。如果 `python3` 指向未安装依赖的系统 Python，请先激活正确环境。macOS Conda 环境如同时加载两份 OpenMP 库，脚本会在加载 PyTorch 前做一次进程级兼容处理。

## 输出与解读

`metrics.json` 包含验证集微调前后结果、最终测试集结果、训练历史，以及验证集六类需求各一条的生成回复。`accuracy` 是模型对较好回复给出更高平均 token 概率的比例；`mean_logp_margin` 是两种回复的平均分差。`risk_phrase_count_*` 只检测明确金额和几种预先列出的风险短语，不能代替人工判断。测试集只在训练完成后用于最终评估；不要根据测试结果反复调参。

本机实测：240 步训练时，抽取的 24 对验证样本由 50% 升至 100%；保存适配器并重新加载后，在完整 48 对测试样本上为 100%（[`evaluation_48.json`](orpo_output/full/evaluation_48.json)）。六类验证问题的生成回复见 [`metrics.json`](orpo_output/full/metrics.json)：微调后能询问缺失信息、避免直接报价或承诺空档，并在合适位置结束。数据和回复模板高度规则化，这些数字只能说明模型学会了本案例的偏好模式，不能代表真实用户提问上的表现。

CPU 上的完整训练耗时取决于机器；快速检查只验证流程，不能作为效果结论。0.5B 模型和规则合成数据适合教学演示，不能直接用于真实服务承诺、报价或排班。真实上线需要人工标注的偏好数据、服务目录及价格接口、排班查询和人工复核。
