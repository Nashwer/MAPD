# 当前复现进度

更新时间：2026-09-24。

## 已完成并验证

### 本地核心流程

- 已建立 MAPD 的数据、检索、搜索环境、MAS、协议质量门、奖励与训练目标模块边界。
- Mock 教师与 fixture BM25 可完成离线协议合成、在线多轨迹 rollout、严格 EM 奖励、PI 选择和联合损失计算。
- 协议 artifact 支持校验、缓存命中、断点续跑以及 JSONL/Parquet veRL 数据导出。
- Windows 轻量本地测试共 31 项，当前全部通过；另有 1 项依赖 PyTorch 的真实
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

### 单卡训练验收（已通过 GPU 验证）

- vLLM rollout 会持久化每个生成 token 的 ID 和 rollout log-prob，schema 会拒绝错位数据。
- replay 根据真实多轮消息历史重建每一轮上下文；检索 observation 进入后续上下文，但不计入
  action-token loss。
- 普通分支与 protocol 特权分支严格使用相同 response token；特权分支通过 `no_grad`
  阻断梯度。
- 已实现逐 token 全词表 reverse-KL、GRPO clipped objective、联合反传、参数更新、
  checkpoint 保存与重新加载。
- 为适配单张 24 GB GPU，验收命令先在独立进程完成 vLLM rollout，释放显存后以
  float32 加载训练模型，并且只训练最后一个 decoder layer。

入口为 `bash mapd.sh train-smoke`。Qwen3-1.7B 单卡实测结果：95 个 distillation
tokens，OPSD loss `1.345361`，总损失 `0.067268`，梯度范数 `0.606946`，抽样参数
变化量约 `1.013e-6`，并成功输出和重载 `checkpoint-step-1/checkpoint.pt`。

该 rollout group 的两个 reward 均为 `0`，因此组相对 advantage 和 GRPO loss 都为
`0`。这不影响本次对 token replay、OPSD、反传、optimizer step 和 checkpoint 的验收，
但还不能算“真实非零 GRPO 更新已验证”。该项留到真实数据与检索接入后，用具有奖励
差异的 rollout group 验收。

### 真实数据与检索代码（已实现，待服务器验收）

- 已实现固定 Hugging Face revision 的 Search-R1 NQ/HotpotQA train/test 下载。
- 已实现 Parquet 流式标准化、规范化问题去重、test/额外 eval 问题隔离，以及固定 seed
  的 25,600 条分层子集；manifest 记录源 revision、各阶段计数和抽样参数。
- 已实现 wiki-18 压缩语料下载、流式 SQLite FTS5 索引构建及原子替换，查询时不把完整
  语料加载到内存。
- 已实现纯命令行 Search-R1 兼容 `/retrieve` HTTP 服务、`/health`、后台状态/日志/停止
  命令，以及真实 QA answer recall@3 smoke。
- 已实现真实 8 路 rollout 中自动寻找非零 reward variance，并在独立进程执行
  GRPO-only optimizer step/checkpoint 的验收入口。

以上代码的 fixture/契约测试已经在本地通过；完整 Parquet、约 5.1 GB 压缩 wiki-18、
全量索引和真实 GPU GRPO 尚未在服务器执行，因此仍标为“待验收”。仓库自带索引使用
SQLite FTS5/BM25；Search-R1 的约 64.6 GB E5 FAISS 资产可下载，但本轮没有把 dense
E5 后端并入默认服务，二者不能混称。

## 尚未完成

- 在服务器实际下载并核验 25,600 条训练数据 manifest。
- 在服务器完成 wiki-18 全量建库并记录体积、耗时和真实 recall@3。
- 将已实现的单卡双分支目标接入 veRL worker；veRL 的标准自定义 policy-loss hook 只提供
  选中 token log-prob，不能直接替代需要全词表 logits 的 OPSD worker 改造。
- 运行已实现的真实 rollout/optimizer 入口，验收非零 GRPO GPU 更新。
- Qwen3-4B、8 GPU、200 steps 的论文规模训练。
- 七个 QA 数据集、五个种子的完整评测与汇总。

## 下次继续时的顺序

1. 在新实例取得本仓库后运行 `bash mapd.sh bootstrap`，验收空环境重建。
2. 若失败，只需保留 `logs/bootstrap-*.log` 和第一个根因报错；把缺失依赖补回仓库脚本。
3. 依次运行 `data-setup`、`wiki-setup`、`retrieval-start`、`retrieval-smoke`，验收真实
   数据和 top-3 检索链路。
4. 运行 `grpo-smoke` 验收非零 GRPO 后，把相同目标接入 veRL worker 并扩展多 GPU 规模。

在第 3 步完成前，CPU mock loss 和真实 vLLM agent smoke 都不能被表述为“已经复现论文
训练结果”。
