# Reproduction Matrix

Status date: 2026-09-24. “Implemented” means the repository contains the
corresponding contract or reference logic. “GPU verified” is reserved for code
that has actually run on the RTX 4090 D test instance. See
[`current-status.md`](current-status.md) for the exact verification snapshot.

| Paper component | Paper setting | Repository status | Fidelity note |
| --- | --- | --- | --- |
| Ephemeral deployment | Reconstruct a usable training host | Repository-pinned bootstrap implemented; component stack verified | First clean-`/home` one-command acceptance run pending |
| Training data | NQ + HotpotQA train, 25,600 examples reported | JSONL artifact and veRL Parquet export implemented | Raw benchmark download/normalization is the next data milestone |
| Evaluation | 7 QA datasets, EM, 5 seeds | Dataset names, strict EM, saved-trajectory aggregation implemented | Benchmark loaders and model inference runner pending |
| Orchestrator | <=4 subtasks, dependency annotations, <=2 rounds | Implemented | Teacher prompt is a reconstruction |
| Searcher | independent agents, <=3 queries each | Implemented | Parallel by ready DAG level |
| Retrieval | wiki-18, top-3 | HTTP contract + fixture BM25 | Full wiki-18 index not present locally |
| Candidate/EM | evidence-only answer then strict EM | Implemented | Answerer no longer receives GT |
| Repair | GT diagnostic, expression/search, <=2 rounds | Implemented | GT is redacted from persisted diagnostics |
| Protocolizer | Succeeded and Evidence variants | Implemented | Invalid raw JSON is retained for audit |
| Quality gate | schema, EM, extractive facts, leak detection | Implemented | Leak check is deterministic answer/alias detection |
| Agentic rollout | <=4 retrieval interaction turns | Environment and vLLM policy adapter implemented; one real two-turn fixture-search trajectory GPU verified | veRL v0.9.1 custom training loop remains GPU-backend work |
| GRPO | group 8, terminal EM reward | Reference objective, veRL reward entry point, and single-device differentiable replay implemented | Qwen GPU acceptance and distributed veRL execution pending |
| OPSD | same current policy, protocol PI, reverse KL | Exact-token dual-context replay, stop-gradient PI branch, and full-vocabulary PyTorch KL implemented | Qwen GPU acceptance and veRL worker integration pending |
| Fallback | failed gate -> correct self-rollout PI | Implemented selector | Availability is decided after online rollouts |
| Joint objective | `L_GRPO + 0.05 L_OPSD` | Scalar reference path tested; real single-device optimizer/checkpoint path implemented | GPU acceptance pending |
| Student models | Qwen3-1.7B and Qwen3-4B | Qwen3-1.7B local vLLM inference GPU verified; both configured | Qwen3-4B and any optimizer update pending |
| Training scale | batch 128, 200 steps, 8 GPUs | Paper-like values configured | Only single-GPU inference has been exercised; training pending |
| Inference | student only, no PI/MAS | Environment contract implemented | Benchmark runner pending |
