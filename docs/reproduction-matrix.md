# Reproduction Matrix

| Paper component | Paper setting | Repository status | Fidelity note |
| --- | --- | --- | --- |
| Training data | NQ + HotpotQA train, 25,600 examples reported | JSONL artifact and veRL Parquet export implemented | Raw benchmark download/normalization is the next data milestone |
| Evaluation | 7 QA datasets, EM, 5 seeds | Dataset names, strict EM, saved-trajectory aggregation implemented | Benchmark loaders and model inference runner pending |
| Orchestrator | <=4 subtasks, dependency annotations, <=2 rounds | Implemented | Teacher prompt is a reconstruction |
| Searcher | independent agents, <=3 queries each | Implemented | Parallel by ready DAG level |
| Retrieval | wiki-18, top-3 | HTTP contract + fixture BM25 | Full wiki-18 index not present locally |
| Candidate/EM | evidence-only answer then strict EM | Implemented | Answerer no longer receives GT |
| Repair | GT diagnostic, expression/search, <=2 rounds | Implemented | GT is redacted from persisted diagnostics |
| Protocolizer | Succeeded and Evidence variants | Implemented | Invalid raw JSON is retained for audit |
| Quality gate | schema, EM, extractive facts, leak detection | Implemented | Leak check is deterministic answer/alias detection |
| Agentic rollout | <=4 retrieval interaction turns | Environment, vLLM policy adapter, and real fixture-search smoke implemented | veRL v0.9.1 custom training loop remains GPU-backend work |
| GRPO | group 8, terminal EM reward | Reference objective and veRL custom reward entry point implemented | Differentiable distributed execution pending |
| OPSD | same current policy, protocol PI, reverse KL | Reference + PyTorch objective implemented | Exact dual-forward/token alignment pending in veRL adapter |
| Fallback | failed gate -> correct self-rollout PI | Implemented selector | Availability is decided after online rollouts |
| Joint objective | `L_GRPO + 0.05 L_OPSD` | Implemented | CPU smoke evaluates complete loss path |
| Student models | Qwen3-1.7B and Qwen3-4B | Qwen3-1.7B local vLLM inference verified; both configured | Qwen3-4B and GPU training pending |
| Training scale | batch 128, 200 steps, 8 GPUs | Configured | Requires Linux GPU environment |
| Inference | student only, no PI/MAS | Environment contract implemented | Benchmark runner pending |
