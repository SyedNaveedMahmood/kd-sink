# CV-0 on bwUniCluster 3.0

This profile runs all four CV-0 components with `Qwen/Qwen3.5-9B` served by a
separate vLLM environment on one NVIDIA H100. The MIRAGE client has its own
environment, installed from this repository's `pyproject.toml`. The model and
AgentDojo revisions and vLLM version are pinned.

## Required cluster layout

Start in your existing `llmrun/` directory:

```bash
cd llmrun
mkdir -p agentrun
git clone --branch CV-0-slurm-job \
  https://github.com/AlZubayer/Mirage-Persist.git \
  agentrun/mirage-persist
mkdir -p agentrun/vllm
cd agentrun/mirage-persist
```

The resulting layout is:

```text
llmrun/
└── agentrun/
    ├── vllm/                  # vLLM venv, caches, model, server logs
    └── mirage-persist/        # this checkout + independent MIRAGE venv
```

The scripts validate this topology and stop with a clear error if the checkout
is elsewhere. No computation or package build is run on a login node.

## 1. Create both environments

```bash
cd llmrun/agentrun/mirage-persist
bash cluster/bwunicluster3/submit_setup.sh
```

Wait for the setup job to finish, then inspect its log:

```bash
squeue -u "$USER"
tail -f cluster/bwunicluster3/logs/mirage-setup-*.out
```

The setup produces:

- `llmrun/agentrun/vllm/.venv` with vLLM `0.25.1`;
- `llmrun/agentrun/mirage-persist/.venv` from `pyproject.toml`;
- `.deps/agentdojo` at revision `089ed468cf3ed0322acc66b0211f26d9d90dbf60`;
- frozen package inventories for both environments.

To run the setup inside an existing CPU allocation instead, use:

```bash
bash cluster/bwunicluster3/setup_all.sh
```

Set `PYTHON_BIN=/path/to/python3.11` if `/usr/bin/python3.11` is not suitable.

## 2. Submit CV-0

```bash
bash cluster/bwunicluster3/submit_cv0.sh
```

This first submits a CPU model-prefetch job, then submits the H100 job with an
`afterok` dependency. If the model snapshot is already complete, skip the
prefetch job with `SKIP_MODEL_PREFETCH=1`.

Common site overrides:

```bash
SBATCH_ACCOUNT=my_project \
SBATCH_GPU_PARTITION=gpu_h100 \
bash cluster/bwunicluster3/submit_cv0.sh
```

The GPU job requests one H100, 16 CPU cores, 128 GiB RAM, and the queue maximum
of 72 hours. It starts vLLM on loopback, waits for `/health`, verifies `/v1/models`,
then runs:

```bash
mirage cv0 run --config configs/cv0/cv0_bwunicluster_qwen3p5_9b.yaml
```

vLLM is configured with Qwen's XML tool-call parser, automatic tool choice,
BF16, a 32K context cap, and non-thinking mode. The latter is also sent on each
MIRAGE request, so the scientific config is not dependent on a server default.

## Outputs and status

- Slurm logs: `cluster/bwunicluster3/logs/`
- vLLM logs/cache/model: `../vllm/logs/` and `../vllm/cache/`
- experiment outputs: `runs/cv0_bwunicluster_qwen3p5_9b/`
- curated artifacts: `artifacts/cv0_bwunicluster_qwen3p5_9b/`

The experiment exit codes are meaningful: `0=PASS`, `1=FAIL`, `2=KILL`. A
scientific FAIL/KILL therefore appears as a non-successful Slurm completion and
must be interpreted using `cv0_verdict.json`, not as an infrastructure failure.

This Qwen-only allocation runs the full CV-0(a-d) scale, but scopes the
family-count gate to one family. It does not replace the preregistered
three-family confirmatory population run.

## Cluster references

- [bwUniCluster 3.0 queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwUniCluster 3.0 GPU job syntax](https://wiki.bwhpc.de/e/BwUniCluster3.0/Slurm#GPU_jobs)
- [bwUniCluster 3.0 filesystems](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture)
- [Qwen3.5-9B model](https://huggingface.co/Qwen/Qwen3.5-9B)
- [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)

## Reduced validation jobs

`configs/cv0/cv0_cluster_quick.yaml` is intended only for short integration
validation. A PASS from this configuration is not a canonical or confirmatory
CV-0 result.

The submission wrapper accepts a configurable GPU wall-time. Example:

    SKIP_MODEL_PREFETCH=1 \
    SBATCH_GPU_PARTITION="dev_gpu_h100,dev_gpu_a100_il,gpu_h100_short,gpu_a100_short,gpu_h100,gpu_h100_il,gpu_a100_il" \
    SBATCH_GPU_TIME="00:30:00" \
    CV0_CONFIG="$PWD/configs/cv0/cv0_cluster_quick.yaml" \
    bash cluster/bwunicluster3/submit_cv0.sh

`SBATCH_GPU_TIME` defaults to `48:00:00` when it is not specified. Development
partitions should only be included when the requested wall-time fits their
limits.
