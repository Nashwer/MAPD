# Shared-server MAPD deployment

This profile keeps retrieval on CPU and reserves explicitly selected GPUs only
for rollout, training, and evaluation. It is intended for the inspected host
with Ubuntu 22.04, CUDA 12.4, no scheduler, no passwordless sudo, and shared
4090/L40S devices.

## Layout

Clone both MAPD and veRL below the large `/data4` filesystem. Do not put model
weights or package caches on the nearly-full root filesystem.

```text
/data4/team2/qjx/MAPD-repro
/data4/team2/qjx/verl
/data4/team2/qjx/models
/data4/team2/qjx/cache
/data4/team2/qjx/tmp
```

Create `.env` from `configs/bupt_shared_server.env.example`, then insert the
teacher API key locally and restrict its permissions:

```bash
cd /data4/team2/qjx/MAPD-repro
cp configs/bupt_shared_server.env.example .env
chmod 600 .env
```

The checked-in profile uses retriever port `28180` because ports `8000` and
`18080` were occupied when the host was inspected. Run the read-only preflight
before installation:

```bash
bash mapd.sh server-preflight
bash mapd.sh bootstrap --skip-model --skip-verify
```

Bootstrap accepts Ubuntu 22.04 and 24.04, discovers an existing CUDA toolkit,
installs a user-owned CPython 3.12 through `uv` for veRL v0.9.1, and uses only
user-writable cache/model directories. The system Python may remain 3.10. If a required system build
tool is absent and passwordless sudo is unavailable, it stops with the exact
administrator package request rather than attempting an interactive sudo.

## Retrieval service

Keep the full compressed E5 retriever on CPU so it remains available while GPU
ownership changes:

```bash
bash mapd.sh retrieval-start
bash mapd.sh retrieval-status
curl -fsS http://127.0.0.1:28180/health
```

All MAPD commands now use `MAPD_RETRIEVER_URL` and
`MAPD_RETRIEVER_HEALTH_URL`; neither the launcher nor health checks assume port
8000.

## GPU jobs

Immediately before a GPU job, inspect current ownership and select only devices
that the server policy permits you to use:

```bash
nvidia-smi
export MAPD_GPU_DEVICES=2
bash mapd.sh server-preflight
```

`MAPD_GPU_DEVICES` contains physical indices and is copied to
`CUDA_VISIBLE_DEVICES`. On a shared server it is mandatory for GPU commands.
The preflight rejects a selected device above the configured memory or
utilization thresholds. Every selected index also receives a non-blocking
per-checkout file lock, so two MAPD jobs from this checkout cannot accidentally
use the same GPU simultaneously.

The guard cannot reserve hardware against other Unix users. Recheck
`nvidia-smi`, follow the administrator's allocation policy, and use only one
homogeneous GPU family in a synchronous training run. Do not combine 4090 and
L40S devices in one data-parallel job.

## Paper-reproduction boundary

This deployment work makes the existing data, protocol, retrieval, smoke
training, and paired evaluation paths portable. It does not turn the current
last-decoder-layer smoke optimizer into the paper's full distributed MAPD
trainer. The version-pinned veRL worker integration remains a separate
implementation and acceptance milestone.
