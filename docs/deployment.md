# Linux deployment without uv

The MAPD repository reuses the already-tested veRL virtual environment. It does
not run `uv sync` and therefore does not mutate veRL's lock-file environment.

With veRL already located at `~/workspace/verl`, deployment and complete
verification are intentionally reduced to two commands:

```bash
git clone --branch dev <YOUR_REPOSITORY_URL> ~/workspace/MAPD-repro
cd ~/workspace/MAPD-repro && bash mapd.sh setup
```

`mapd.sh` automatically discovers the sibling environment at
`~/workspace/verl/.venv`. Only set `VERL_VENV` when the environment is stored
elsewhere.

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
