# Reproducible Linux GPU deployment

## One-command rebuild for ephemeral instances

After the MAPD repository is present, rebuild or repair the complete server
environment with one command:

```bash
cd ~/workspace/MAPD-repro && bash mapd.sh bootstrap
```

No persistent `/home` directory is assumed. The minimum base-image contract is
Ubuntu 24.04 x86_64, a working NVIDIA driver/GPU, Bash, `apt`, root access (or
`sudo`), and a checked-out copy of this repository. The bootstrap verifies that contract and
then obtains every other system package, Python package, source checkout, and
model required by the current repository revision.

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
