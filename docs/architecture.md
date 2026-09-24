# MAPD Reproduction Architecture

## 1. Paper-level end-to-end flow

```text
OFFLINE PROTOCOL SYNTHESIS (proprietary teacher; cached once)

NQ + HotpotQA train samples
          |
          v
  Orchestrator -- dependency DAG, <=4 subtasks, <=2 rounds
          |
          +--> Searcher per subtask -- <=3 queries --> wiki-18 top-3
          |          |
          |          +--> concise finding + passage ids
          |
          +--> Answerer (never sees GT) --> candidate --> strict EM
                                                |
                                     fail ------+------ success
                                       |                 |
                           Repair sees GT only           |
                           <=2 diagnostic rounds         |
                                       \_________________/
                                                |
                                    complete exploration log
                                                |
                                         Protocolizer
                                  /                           \
                         Succeeded Protocol             Evidence Protocol
                         answer + grounded=true         answer=null + partial
                                  \                           /
                           schema / EM / extractive grounding / leak gate
                                                |
                                      accepted protocol cache

ONLINE JOINT TRAINING (open-source student; protocol never enters rollout)

question x --> current student + retrieval environment --> G trajectories y
                                                        --> terminal EM rewards
                                                               |
                                  +----------------------------+------------------+
                                  |                                               |
                   student branch pi(.|x,y<t)                    PI selection
                                  |                              protocol if valid;
                                  |                              else correct self-rollout
                                  |                                               |
                                  |                         privileged branch pi(.|x,p,y<t)
                                  |                         same current policy, stop-gradient
                                  |                                               |
                          clipped GRPO loss               token-level KL(student || privileged)
                                  \______________________________  _______________/
                                                                 \/
                                                    L = L_GRPO + lambda * L_OPSD
                                                                 |
                                                        update current student

EVALUATION

single student policy + retrieval, no teacher and no protocol
--> NQ / TriviaQA / PopQA / HotpotQA / 2Wiki / MuSiQue / Bamboogle
--> exact-match success rate, five evaluation seeds
```

The proprietary model participates only in offline synthesis. Ground truth is available to strict EM, the
Repair diagnostic call, and reward computation; it is not provided to Orchestrator, Searcher, Answerer,
student rollout, or retrieval queries.

## 2. Implemented module boundaries

| Module | Responsibility | Stable boundary |
| --- | --- | --- |
| `config` | paper parameters and explicitly marked reproduction choices | `AppConfig` |
| `data` | QA schema and veRL record conversion | `QASample` / `to_verl_record()` |
| `retrieval` | local BM25 or Search-R1-compatible HTTP lookup | `Retriever.search()` |
| `environment` | action parser and multi-turn student search environment | `AgenticSearchEnvironment.rollout()` |
| `mas` | Orchestrator, Searcher, Repair, Protocolizer and Stage A/B/C coordinator | `MASPipeline.synthesize()` |
| `protocol` | JSON schema, leak check, grounding check and admission gate | `validate_protocol()` |
| `trainer.grpo` | group advantages and clipped policy objective | `grpo_loss()` |
| `trainer.opsd` | PI selection, sequence alignment and reverse KL | `torch_reverse_kl()` |
| `trainer.mapd_trainer` | joint loss and online backend contract | `MAPDTrainingLoop` |
| `reward` | normalized exact-match terminal reward | `exact_match()` |
| `evaluation` | benchmark-level QA aggregation | `evaluate_trajectories()` |

The lightweight core deliberately has no CUDA imports. A separate vLLM policy adapter now implements real
single-GPU generation and has been verified with Qwen3-1.7B. The production training backend must still
implement the backend contract with veRL while the data, environment, and objective invariants remain
testable on Windows.

## 3. Token and gradient contract

For every sampled multi-turn trajectory, the backend must retain:

1. the old-policy log probability of each generated action token;
2. current student logits under `(x, y<t)`;
3. current privileged logits under `(x, p, y<t)` for the exact same `y`;
4. reference-policy KL values used by GRPO;
5. a mask that is true only for model-generated tokens and false for prompts, padding, and retrieval observations.

The privileged logits are detached. OPSD is the exact full-vocabulary reverse KL from the student distribution
to the privileged distribution, averaged over active response tokens. It is not a sampled-token proxy in the
reference implementation.

## 4. Artifact contract

Each synthesis run writes:

```text
artifacts.jsonl      complete SynthesisArtifact records
explorations.jsonl   subtask DAGs, queries, passages, findings and repair audit
protocols.jsonl      validated protocol or rejected raw protocol
quality.jsonl        six deterministic admission checks
training.jsonl       veRL-style prompt/reward/PI records
manifest.json        schema version, config hash, counts and cache hits
online_smoke.json    CPU mock rollout/reward/PI/loss integration trace
```

Malformed protocol JSON is retained as `raw_protocol`, marked as a schema failure, and routed to online
self-rollout fallback rather than crashing synthesis.

## 5. Verified execution boundary

The current GPU smoke test exercises this concrete path:

```text
Qwen3-1.7B/vLLM -> <search> -> fixture BM25 -> <information>
                 -> Qwen3-1.7B/vLLM -> <answer> -> strict EM reward
```

It verifies inference-time state transitions and artifact serialization, but it does not collect training
log probabilities, replay tokens through the privileged branch, backpropagate a joint objective, or update
model weights. Full wiki-18 retrieval is also outside this already-verified smoke boundary.

The separate `train-smoke` acceptance path now implements those missing single-device mechanics in two
processes: vLLM first records exact response token ids and rollout log-probabilities; after vLLM exits, a
Transformers model replays the same tokens under normal and protocol-conditioned contexts, performs the
joint backward pass, updates the last decoder layer, saves a checkpoint, and reloads it. This path is
implemented but remains pending real Qwen GPU verification.

## 6. What remains backend-specific

The paper does not publish its role prompts, GRPO clipping coefficient, reference-KL coefficient, five seed
values, tokenizer alignment code, or trainer patch. This repository has not integrated an authoritative
trainer implementation for those missing details, so they cannot honestly be called exact reproduction yet.
They are isolated behind configuration and `OnPolicyTrainingBackend` so that authoritative code or
experimentally verified choices can replace them without changing the offline artifacts.

The next acceptance milestone is to run `bash mapd.sh train-smoke` on Qwen3-1.7B. Once it passes, the next
implementation milestone is moving the same full-vocabulary dual-forward objective into a veRL worker. A
standard veRL custom policy-loss hook is insufficient because it receives selected-token log-probabilities,
whereas MAPD OPSD requires both branches' complete vocabulary logits.
