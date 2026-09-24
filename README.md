# MAPD Reproduction

本项目复现论文 **From Proprietary to Open-Source: Bridging the Distribution Gap via Multi-Agent Protocol Distillation in Agentic Search** 的核心流程。

当前版本提供一个可在 Windows/CPU 上验证、并能接入 Linux GPU 后端的全流程架构：

- 结构化 MAPD protocol 与训练样本数据契约
- Mock / OpenAI-compatible 教师接口
- 内存 BM25 / HTTP 检索接口
- Search-R1 NQ/HotpotQA 流式标准化、去重、held-out 隔离与 25,600 条子集构建
- 可持久化的 wiki-18 SQLite FTS5 索引与 Search-R1 兼容 HTTP 服务
- 依赖感知 Orchestrator、独立多查询 Searcher、Answerer、Repair、Protocolizer
- Schema、EM、抽取式 grounding、答案泄漏质量门
- student 多轮搜索环境、G 路 rollout、EM reward 和 self-rollout fallback
- Qwen3-1.7B/vLLM 真实 GPU 搜索轨迹验证
- GRPO、完整词表 OPSD reverse-KL、stop-gradient privileged branch 的参考实现
- 可重复、可断点续跑的 JSONL artifacts
- 无网络、无 API、无 GPU 的端到端 smoke test

## 当前进度

截至 2026-09-24，轻量核心的 29 项本地测试全部通过（另 1 项无 Torch 时跳过）；Qwen3-1.7B 已在 RTX
4090 D 上通过真实 vLLM 推理和两轮 agent 搜索验证。真实轨迹完成了
`<search>` → BM25 `<information>` → `<answer>`，严格 EM reward 为 `1.0`。
单卡 `train-smoke` 也已完成真实 Qwen3-1.7B 双上下文反传、参数更新和
checkpoint reload；该次 rollout group 的 reward 全为 0，因此 GPU 运行中的 GRPO
贡献为 0，OPSD 路径与联合优化器闭环已得到验证。

仓库已经包含面向一天期 GPU 实例的一键重建脚本，但最新脚本仍需在下一台
`/home` 全空的新实例上做首次端到端验收。单卡双上下文 token replay、全词表
OPSD、一次 optimizer step 和 checkpoint reload 的 `train-smoke` 已通过 GPU 验收；
真实数据下载/标准化、完整 wiki-18 建库、top-3 检索和非零组内 reward variance 的
GRPO 验收入口已实现，但还要在服务器下载大文件并实际运行。分布式 veRL worker、
论文规模训练和七数据集评测尚未完成。
不能将任一 smoke 通过等同于论文结果复现。

详细的完成项、验证环境、待办边界和下次继续顺序见
[`docs/current-status.md`](docs/current-status.md)。

## 目录结构

```text
MAPD-Reproduction/
├── data/                       # 数据和本地索引，不放 Python 业务逻辑
│   ├── nq/
│   ├── hotpotqa/
│   ├── protocols/
│   └── wiki18/index/
├── src/mapd/
│   ├── retrieval/              # wiki-18 与远程 retriever client
│   ├── environment/            # student 搜索环境与动作 parser
│   ├── mas/                    # orchestrator/searcher/repair/protocolizer
│   ├── protocol/               # schema/validator/leak/grounding
│   ├── trainer/                # grpo/opsd/mapd_trainer
│   ├── reward/                 # exact match reward
│   ├── evaluation/             # QA evaluation
│   └── data/                   # QA schema 与 veRL record 转换
├── scripts/                    # 可执行入口，不承载核心逻辑
├── configs/
└── tests/
```

Python 代码保留在 `mapd` 命名空间内，避免安装后与第三方 `environment`、`protocol`
等顶层包重名。目录职责与参考结构一致，但数据目录和源码目录不会混在一起。

## 本机启动

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\mapd.exe doctor --config configs/local_smoke.yaml
.\.venv\Scripts\mapd.exe smoke --config configs/local_smoke.yaml
.\.venv\Scripts\python.exe -m pytest
```

也可以始终通过 Python 模块运行，避免 shell 激活状态差异：

```powershell
.\.venv\Scripts\python.exe -m mapd smoke --config configs/local_smoke.yaml
```

## 主要命令

```text
mapd doctor        检查配置、Python、语料和教师 API 环境
mapd smoke         使用 fixture 跑通离线合成 + 在线 rollout/reward/PI/loss
mapd synthesize    对 QA JSONL 执行协议合成与质量门
mapd validate      重新验证已有协议 artifact
mapd prepare-data  转换为 veRL 风格训练 JSONL
mapd evaluate-trajectories  汇总已保存轨迹的严格 EM
```

真实 API 配置参考 `.env.example`。密钥、数据集、Wikipedia 索引、运行产物、模型和 checkpoint 不进入 Git。

完整流程和模块边界见 [`docs/architecture.md`](docs/architecture.md)，论文逐项覆盖情况见
[`docs/reproduction-matrix.md`](docs/reproduction-matrix.md)。

Linux GPU 服务器的可重复部署步骤见
[`docs/deployment.md`](docs/deployment.md)。
其中包含纯命令行后台任务管理：`start/status/logs/follow/stop/jobs`，不依赖图形界面。

一天期 GPU 实例的完整重建、模型复用与验证只需：

```bash
cd ~/workspace/MAPD-repro && bash mapd.sh bootstrap
```

该命令不假定 `/home` 会被保留。脚本会明确报告虚拟环境是 `MISSING`、
`INCOMPLETE`、`BROKEN` 还是 `HEALTHY`；前三种情况自动按仓库中的固定版本
清单重建或修复，已有缓存仅用于加速。版本入口为
`scripts/bootstrap_versions.env`，附加 Python 依赖位于
`requirements/server-bootstrap.txt`。

部署完成后，下一阶段的单卡训练闭环验收使用：

```bash
bash mapd.sh train-smoke
```

它分两个进程执行：先用 vLLM 保存带精确 token/log-prob 的 rollout 并退出释放显存，
再用同一 Qwen checkpoint 重放普通/特权上下文，只训练最后一个 decoder layer，执行一次
MAPD 联合反传，并验证 checkpoint 能恢复。该命令是低显存工程验收，不是论文的 8 GPU
正式训练配置。

## 真实数据、wiki-18 与非零 GRPO 验收

以下命令均不需要手工编辑文件：

```bash
bash mapd.sh data-setup
bash mapd.sh wiki-setup
bash mapd.sh retrieval-start
bash mapd.sh retrieval-status
bash mapd.sh retrieval-smoke 20
bash mapd.sh grpo-smoke
```

`data-setup` 下载固定 revision 的 Search-R1 NQ/HotpotQA Parquet，使用其 test split
建立 held-out 问题集合，按规范化问题去重并排除重叠，然后以固定 seed 构建 25,600 条
训练子集。论文未公开这 25,600 条的抽样 ID，因此当前实现明确采用 NQ/HotpotQA
各 12,800 条的可复现分层抽样。

`wiki-setup` 下载约 5.1 GB 的压缩 wiki-18 语料并构建磁盘持久化 SQLite FTS5/BM25
索引；这是仓库自带、低依赖的复现后端，不是论文/Search-R1 的 E5 dense index。
`grpo-smoke` 会在真实 QA 和 top-3 检索上寻找同时包含 0/1 reward 的 8 路 rollout
group，再在第二个进程执行一次 GRPO-only 更新并保存 checkpoint。


## 后续训练路线

正式训练在 Linux GPU 环境中接入 veRL：student 使用普通上下文进行多轮检索 rollout，同一步使用当前 actor 权重在 protocol 特权上下文中进行 stop-gradient 打分，再优化 `L_GRPO + lambda_opsd * L_OPSD`。论文配置保存在 `configs/paper_like.yaml`。当前仓库已经实现框架无关参考公式和 PyTorch 目标函数；分布式 Qwen/veRL backend 是下一模块，不能用 CPU mock 的通过来替代真实训练结论。
