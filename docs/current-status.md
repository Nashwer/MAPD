# 当前复现进度

更新时间：2026-09-24。

## 已完成并验证

### 本地核心流程

- 已建立 MAPD 的数据、检索、搜索环境、MAS、协议质量门、奖励与训练目标模块边界。
- Mock 教师与 fixture BM25 可完成离线协议合成、在线多轨迹 rollout、严格 EM 奖励、PI 选择和联合损失计算。
- 协议 artifact 支持校验、缓存命中、断点续跑以及 JSONL/Parquet veRL 数据导出。
- Windows 轻量本地测试共 22 项，当前全部通过；另有 1 项依赖 PyTorch 的真实
  optimizer/checkpoint 测试会在服务器环境执行，本机无 Torch 时跳过。

### Linux GPU 推理链路

已在以下服务器环境完成真实验证：

| 项目 | 已验证值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D（24 GB） |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.24.0 |
| NumPy | 2.3.5 |
| CUDA toolkit | 13.0，包含 nvcc 和 cuRAND headers |
| 模型 | Qwen3-1.7B，本地目录加载 |

真实检查结果：

1. `model-smoke` 成功加载 Qwen3-1.7B，并通过 vLLM 完成一次 GPU 生成。
2. `agent-smoke` 成功完成两轮轨迹：模型先输出 `<search>`，fixture BM25 返回
   `<information>`，模型再输出最小答案 `<answer>Alan Rickman</answer>`。
3. 该轨迹严格 exact-match reward 为 `1.0`，并写入
   `artifacts/agent_smoke/trajectory.jsonl`。

这证明当前的模型加载、动作解析、检索观察回填、终止答案解析和奖励链路可以在真实
GPU 上连通。它还不是论文规模训练或完整 wiki-18 评测。

### 临时实例重建

- `bash mapd.sh bootstrap` 已实现仓库驱动的一键环境创建/修复。
- 不假定 `/home` 持久化；缓存和已有模型只作为可选加速。
- 虚拟环境会报告 `MISSING`、`INCOMPLETE`、`BROKEN` 或 `HEALTHY`。
- 系统依赖、CUDA 编译组件、veRL 环境、额外 Python 依赖、模型下载、MAPD
  安装与 smoke 验证均已编排进脚本。
- 固定版本入口为 `scripts/bootstrap_versions.env`，附加 Python 依赖入口为
  `requirements/server-bootstrap.txt`。

注意：这些依赖已在现有服务器上逐项安装并验证，但最新的一键脚本还没有在一台
`/home` 全空的新实例上完成首次端到端验收。这是下一台实例启动后的第一项任务。

### 单卡训练验收代码（待 GPU 验证）

- vLLM rollout 会持久化每个生成 token 的 ID 和 rollout log-prob，schema 会拒绝错位数据。
- replay 根据真实多轮消息历史重建每一轮上下文；检索 observation 进入后续上下文，但不计入
  action-token loss。
- 普通分支与 protocol 特权分支严格使用相同 response token；特权分支通过 `no_grad`
  阻断梯度。
- 已实现逐 token 全词表 reverse-KL、GRPO clipped objective、联合反传、参数更新、
  checkpoint 保存与重新加载。
- 为适配单张 24 GB GPU，验收命令先在独立进程完成 vLLM rollout，释放显存后以
  float32 加载训练模型，并且只训练最后一个 decoder layer。

入口为 `bash mapd.sh train-smoke`。这部分代码尚未在服务器运行，因此目前只能标记为
“实现、待 GPU 验收”，不能标记为训练闭环已通过。

## 尚未完成

- 从原始 NQ、HotpotQA 构建论文规模的 25,600 条训练数据。
- 部署完整 wiki-18 索引和生产检索服务。
- 将已实现的单卡双分支目标接入 veRL worker；veRL 的标准自定义 policy-loss hook 只提供
  选中 token log-prob，不能直接替代需要全词表 logits 的 OPSD worker 改造。
- 在真实 Qwen3-1.7B 上验收一次可恢复的 optimizer step 与 checkpoint reload。
- Qwen3-4B、8 GPU、200 steps 的论文规模训练。
- 七个 QA 数据集、五个种子的完整评测与汇总。

## 下次继续时的顺序

1. 在新实例取得本仓库后运行 `bash mapd.sh bootstrap`，验收空环境重建。
2. 若失败，只需保留 `logs/bootstrap-*.log` 和第一个根因报错；把缺失依赖补回仓库脚本。
3. 环境通过后运行 `bash mapd.sh train-smoke`，验收真实 rollout、双分支 token replay、
   optimizer step 与 checkpoint reload。
4. 单卡闭环通过后，把相同目标接入 veRL worker，再扩展数据、wiki-18 和多 GPU 规模。

在第 3 步完成前，CPU mock loss 和真实 vLLM agent smoke 都不能被表述为“已经复现论文
训练结果”。
