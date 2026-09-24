# Reproducible Linux GPU deployment

## One-command rebuild for ephemeral instances

After the MAPD repository is present, rebuild or repair the complete server
environment with one command:

```bash
cd ~/workspace/MAPD-repro && bash mapd.sh bootstrap
```

No persistent `/home` directory is assumed. The minimum base-image contract is
Ubuntu 24.04 x86_64, a working NVIDIA driver/GPU, Bash, `apt`, root access (or
`sudo`), and a checked-out copy of this repository. The bootstrap verifies that
contract and then obtains every other system package, Python package, source
checkout, and model required by the current repository revision. Network access
to Ubuntu/NVIDIA package repositories, GitHub, Python package indexes, and the
configured Hugging Face endpoint is required for an empty machine.

The bootstrap is idempotent. It classifies the virtual environment as
`MISSING`, `INCOMPLETE`, `BROKEN`, or `HEALTHY`. A missing or damaged
`~/workspace/verl/.venv` is created/repaired from veRL's frozen lock file; a
healthy one is reused. The model under `~/models/Qwen3-1.7B` is handled the
same way. It then installs MAPD, runs offline tests and a real agent trajectory,
and writes an exact environment record to
`artifacts/environment/manifest.json`. Its full terminal log is retained under
`logs/bootstrap-*.log`.

On a completely blank home directory, obtain MAPD first and then run the same
bootstrap command. The script clones the repository-pinned veRL checkout when
the sibling `~/workspace/verl` checkout is absent. If `/home` happens to be
persistent, its environment, package caches, and model are optional speedups;
correctness never depends on them.

For a dependency-only rebuild without downloading/loading the model:

```bash
bash mapd.sh bootstrap --skip-model --skip-verify
```

The maintained bootstrap contract currently includes Python 3.12 headers,
Ninja, the CUDA 13.0 compiler, cuRAND headers, veRL FSDP + vLLM, NumPy 2.3.5,
PyArrow, pytest, MAPD, and Qwen3-1.7B. Pinned stack inputs live in
`scripts/bootstrap_versions.env`; MAPD-side Python additions live in
`requirements/server-bootstrap.txt`. Future modules must update these files
and the bootstrap script in the same code change, so the one-command rebuild
remains complete.

## Current verification baseline

The component stack was verified on an NVIDIA GeForce RTX 4090 D (24 GB) with
Python 3.12.3, PyTorch 2.11.0+cu130, vLLM 0.24.0, NumPy 2.3.5, and the CUDA
13.0 compiler/toolkit. Both the standalone model generation and the real
Qwen3-1.7B -> fixture BM25 -> Qwen3-1.7B agent trajectory passed. The latter
ended with strict exact-match reward `1.0` and `AGENT SMOKE OK`.

The dependencies were discovered and verified on the current server before the
bootstrap was consolidated. The consolidated script itself still needs its
first acceptance run on a new instance with an empty `/home`. After that run,
success requires all of the following:

```text
Virtual environment: HEALTHY (or a successful MISSING -> creation path)
all tests passed (the Torch-dependent test may run instead of skip)
AGENT SMOKE OK
BOOTSTRAP OK
```

The exact package/Git/CUDA record from each successful instance is written to
`artifacts/environment/manifest.json`. Do not diagnose future instances from
screenshots of the full traceback; retain `logs/bootstrap-*.log` and extract the
first root-cause error instead.

## Reuse an existing veRL environment

The lighter `setup` command reuses an already-tested veRL virtual environment.
It does not run `uv sync` and therefore does not mutate that environment.

With veRL already located at `~/workspace/verl`, deployment and complete
verification are intentionally reduced to two commands:

```bash
git clone --branch dev <YOUR_REPOSITORY_URL> ~/workspace/MAPD-repro
cd ~/workspace/MAPD-repro && bash mapd.sh setup
```

`mapd.sh` automatically discovers the sibling environment at
`~/workspace/verl/.venv`. Only set `VERL_VENV` when the environment is stored
elsewhere.

The GPU smoke commands require the following Ubuntu packages in addition to
the veRL environment: `python3.12-dev`, `ninja-build`,
`cuda-compiler-13-0`, and `libcurand-dev-13-0`. The launcher automatically
uses `/usr/local/cuda-13.0` when it is installed there.

## Local model smoke test

After downloading Qwen3-1.7B to `~/models/Qwen3-1.7B`, load it with vLLM and
run one real GPU generation using a short command:

```bash
bash mapd.sh model-smoke
```

For a different location, pass the directory explicitly:

```bash
bash mapd.sh model-smoke /path/to/Qwen3-1.7B
```

Run a real multi-turn agent check with the local fixture retriever:

```bash
bash mapd.sh agent-smoke
```

This command requires the model to emit a search action, executes BM25, feeds
the retrieved observation back to the same model, checks the terminal answer,
and writes the complete trajectory to
`artifacts/agent_smoke/trajectory.jsonl`.

## Headless job control

No desktop session, notebook, W&B, or web dashboard is required. Start a job
under `nohup` with a stable name:

```bash
bash mapd.sh start
```

The SSH connection may then be closed. Inspect it later with:

```bash
bash mapd.sh status
bash mapd.sh logs
bash mapd.sh follow
```

`follow` streams the log and can be left with `Ctrl-C` without stopping the
job. To request a graceful stop:

```bash
bash mapd.sh stop
```

List all known jobs and inspect synthesis artifacts directly:

```bash
bash mapd.sh jobs
bash mapd.sh status
```

State is stored under `.run/`, logs under `logs/`, and both are ignored by
Git. Every synthesis progress boundary rewrites a valid `artifacts.jsonl`
checkpoint, so a later invocation can reuse completed examples.

The install script uses only:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

For training data, veRL should receive Parquet rather than the JSONL audit
artifact:

```bash
python -m pip install "pyarrow>=17,<22"
python -m mapd prepare-data \
  --artifacts artifacts/smoke/artifacts.jsonl \
  --output data/protocols/train_smoke.parquet
```

`mapd.reward.verl_adapter:compute_score` is the custom strict-EM reward entry
point for veRL. The real MAPD optimizer still requires a version-pinned veRL
adapter that performs a second, stop-gradient forward pass under privileged
protocol context. The CPU smoke objective is an invariant test, not a claim
that distributed MAPD training is already implemented.

## Single-GPU optimizer acceptance

After bootstrap and agent smoke succeed, run the next acceptance stage:

```bash
bash mapd.sh train-smoke > train-smoke.log 2>&1
tail -n 80 train-smoke.log
```

The command deliberately uses two processes so the vLLM engine has released
GPU memory before the differentiable model is loaded. It records exact rollout
token ids/log-probabilities, replays the same response tokens under student and
protocol-conditioned contexts, performs full-vocabulary reverse KL with the
privileged branch detached, updates the final decoder layer, and reloads the
saved checkpoint.

Successful acceptance ends with:

```text
TRAINING ROLLOUT OK
MAPD OPTIMIZER SMOKE OK
```

Artifacts are written under `artifacts/train_smoke/`. This is intentionally a
single-GPU, one-step engineering test. It does not claim fidelity to the
paper's full-parameter, 8-GPU, 200-step run, and it does not yet use veRL's
distributed actor worker.

## Real data, wiki-18, and GRPO acceptance

After bootstrap, prepare the version-pinned Search-R1 NQ/HotpotQA data:

```bash
set -o pipefail
bash mapd.sh data-setup 2>&1 | tee data-setup.log
```

Success ends with `REAL DATA PREP OK`. Inspect the auditable counts without an
editor:

```bash
cat data/training/manifest.json
wc -l data/training/mapd_train_25600.jsonl
```

Download the compressed wiki-18 corpus and build the persistent sparse index.
This is the longest CPU/disk stage and may consume a substantial part of a
one-day instance:

```bash
bash mapd.sh wiki-start
bash mapd.sh wiki-status
bash mapd.sh wiki-logs
```

The SSH session may be closed after `wiki-start`; re-run `wiki-status` and
`wiki-logs` later. Success ends with `WIKI18 INDEX BUILD OK` (or
`WIKI18 INDEX REUSED`). The corpus download is about 5.1 GB;
the final SQLite index is larger, so check free disk first with `df -h .`.
Start and inspect the headless service:

```bash
bash mapd.sh retrieval-start
bash mapd.sh retrieval-status
bash mapd.sh retrieval-logs
bash mapd.sh retrieval-smoke 20
```

The smoke command performs real top-3 requests and prints answer recall plus
document IDs. It accepts recall zero by default because this portable FTS5/BM25
backend is not the paper's E5 dense index; the purpose of the first run is to
verify the full corpus/API path and measure the actual baseline.

Finally, perform the nonzero-GRPO acceptance:

```bash
bash mapd.sh grpo-smoke > grpo-smoke.log 2>&1
tail -n 100 grpo-smoke.log
```

It searches up to 12 real questions for an 8-rollout group containing both
reward 0 and reward 1, releases vLLM, then executes a GRPO-only optimizer step.
Success requires both `NONZERO REWARD VARIANCE ROLLOUT OK` and
`REAL GRPO OPTIMIZER SMOKE OK`. If no mixed group is found, run the rollout
script directly with a larger `--candidate-limit`; this is a sampling outcome,
not proof of a broken optimizer. At the first replay step the printed scalar
GRPO loss may still cancel to zero because normalized group advantages sum to
zero at policy ratio 1; acceptance therefore checks mixed rewards, nonzero
gradient norm, a changed parameter, and a saved checkpoint.

## Budget-capped DeepSeek protocol smoke

The repository root `.env` is ignored by Git and is loaded automatically by
`mapd.sh`. Create it without using an editor and without placing the secret in
shell history:

```bash
cd ~/workspace/MAPD-repro
read -rsp "DeepSeek API key: " MAPD_KEY && echo
printf '%s\n' \
  'MAPD_TEACHER_BASE_URL=https://api.deepseek.com' \
  "MAPD_TEACHER_API_KEY=$MAPD_KEY" \
  'MAPD_TEACHER_MODEL=deepseek-flash' \
  'MAPD_TEACHER_BUDGET_USD=5.0' \
  'MAPD_RETRIEVER_URL=http://127.0.0.1:8000/retrieve' > .env
unset MAPD_KEY
chmod 600 .env
```

The template selects `deepseek-flash`, disables thinking, caps each response at
2,048 tokens, and applies a conservative USD 5 budget using peak-hour
cache-miss prices. Keep the retrieval service running and generate 20 real
protocols with:

```bash
bash mapd.sh retrieval-start
bash mapd.sh protocol-smoke 20 2>&1 | tee protocol-smoke.log
```

Every completed sample checkpoints immediately. Per-request token and estimated
cost records are appended to
`artifacts/protocol_smoke/teacher_usage.jsonl`; the final aggregate is stored in
`artifacts/protocol_smoke/manifest.json`. A repository-wide ledger at
`artifacts/teacher_usage.jsonl` makes the USD 5 guard cumulative across all
shards on the instance instead of resetting per output directory. Re-running
the same command reuses valid artifacts. Inspect results without exposing the
key:

```bash
tail -n 5 artifacts/protocol_smoke/teacher_usage.jsonl
cat artifacts/protocol_smoke/manifest.json
wc -l artifacts/protocol_smoke/artifacts.jsonl
```

## Portable protocol shards for expiring instances

Protocol synthesis uses the remote teacher API and the retrieval service but
does not use the GPU. Split work by the stable position in
`mapd_train_25600.jsonl`; never assign overlapping ranges to different
instances. For example, run examples 0-99 as a background job:

```bash
bash mapd.sh protocol-start 0 100
bash mapd.sh protocol-status
bash mapd.sh protocol-logs
```

Each completed example atomically updates `artifacts.jsonl`. A bundle may be
exported after completion or while the job is still running; only the last
complete checkpoint is included:

```bash
bash mapd.sh protocol-export 0 100
```

The command writes `exports/protocol_offset_0_count_100.tar.gz` and prints the
archive SHA-256. Download that single file from the instance. It contains no
`.env` file or API key. On another checkout, upload the archive and restore the
same shard directory before resuming the same command:

```bash
mkdir -p incoming
# Upload protocol_offset_0_count_100.tar.gz into incoming/ using the platform UI or scp.
bash mapd.sh protocol-restore incoming/protocol_offset_0_count_100.tar.gz
bash mapd.sh protocol-start 0 100
```

Completed sample IDs are cache hits, so only unfinished examples call the
teacher. Independent instances can instead process `0 100`, `100 100`,
`200 100`, and so on. Merge all downloaded bundles into a de-duplicated,
conflict-checked training artifact:

```bash
bash mapd.sh protocol-merge incoming/protocol_offset_*.tar.gz
cat artifacts/protocol_merged/manifest.json
```

The bundle records per-file checksums, source range, completed IDs, quality
count, and usage log. Import refuses checksum mismatches and restore refuses to
overwrite a different local checkpoint. Restore also de-duplicates its usage
records into `artifacts/teacher_usage.jsonl`; restore all earlier bundles on a
new instance before starting more paid synthesis so the budget guard includes
the previous spend.
