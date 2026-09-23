# MAPD Reproduction

本项目复现论文 **From Proprietary to Open-Source: Bridging the Distribution Gap via Multi-Agent Protocol Distillation in Agentic Search** 的核心流程。

当前版本提供一个可在 Windows/CPU 上验证、并能接入 Linux GPU 后端的全流程架构：

- 结构化 MAPD protocol 与训练样本数据契约
- Mock / OpenAI-compatible 教师接口
- 内存 BM25 / HTTP 检索接口
- 依赖感知 Orchestrator、独立多查询 Searcher、Answerer、Repair、Protocolizer
- Schema、EM、抽取式 grounding、答案泄漏质量门
- student 多轮搜索环境、G 路 rollout、EM reward 和 self-rollout fallback
- GRPO、完整词表 OPSD reverse-KL、stop-gradient privileged branch 的参考实现
- 可重复、可断点续跑的 JSONL artifacts
- 无网络、无 API、无 GPU 的端到端 smoke test

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
```

真实 API 配置参考 `.env.example`。密钥、数据集、Wikipedia 索引、运行产物、模型和 checkpoint 不进入 Git。

完整流程和模块边界见 [`docs/architecture.md`](docs/architecture.md)，论文逐项覆盖情况见
[`docs/reproduction-matrix.md`](docs/reproduction-matrix.md)。

Linux GPU 服务器复用现有 veRL 虚拟环境、且完全绕过 uv 的部署步骤见
[`docs/deployment.md`](docs/deployment.md)。
其中包含纯命令行后台任务管理：`start/status/logs/follow/stop/jobs`，不依赖图形界面。

服务器首次验证只需：

```bash
cd ~/workspace/MAPD-repro && bash mapd.sh setup
```


## 后续训练路线

正式训练在 Linux GPU 环境中接入 veRL：student 使用普通上下文进行多轮检索 rollout，同一步使用当前 actor 权重在 protocol 特权上下文中进行 stop-gradient 打分，再优化 `L_GRPO + lambda_opsd * L_OPSD`。论文配置保存在 `configs/paper_like.yaml`。当前仓库已经实现框架无关参考公式和 PyTorch 目标函数；分布式 Qwen/veRL backend 是下一模块，不能用 CPU mock 的通过来替代真实训练结论。
