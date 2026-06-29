# Workstation setup runbook

Target workstation: local Linux box with NVIDIA RTX 5090.

## 1. System prerequisites

Install:

- recent NVIDIA driver supporting CUDA 12.9+ runtime,
- Git + Git LFS,
- uv,
- build-essential / compiler toolchain,
- Vulkan-capable NVIDIA driver/runtime for later ManiSkill visual observations,
- optional: tmux, htop, nvtop, ffmpeg.

## 2. Clone repo

```bash
git clone --recurse-submodules git@github.com:YOUR_ORG/pg3d.git
cd pg3d
```

If submodules were not cloned:

```bash
git submodule update --init --recursive
```

## 3. Python environment

```bash
uv sync --extra cu129 --group dev
make gpu-check
```

Expected for RTX 5090:

- `torch.cuda.is_available()` is true,
- device name includes RTX 5090,
- capability should be Blackwell/sm_120 class,
- matrix multiply smoke test succeeds.

If this fails, do not debug DP3 first. Fix PyTorch/CUDA installation first.

Then verify the pg3d-native DP3 policy smoke:

```bash
uv run python scripts/smoke_dp3_policy.py --device cuda
```

## 4. ManiSkill

Install the simulator optional extra:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

Optionally set an asset directory:

```bash
export MS_ASSET_DIR=/path/to/maniskill_assets
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```

Then follow `docs/runbooks/maniskill_setup.md`.

## 5. Verify repo

```bash
make test
make lint
make smoke
```

`make lint` intentionally excludes `external/`; submodules are not linted as pg3d source.

## 6. Verify W&B

```bash
wandb login
# or:
export WANDB_MODE=offline
```

## Notes

- If ManiSkill/SAPIEN fails under Python 3.11, document the failure in `docs/status.md` and create an ADR before downgrading to Python 3.10 or splitting environments.
- Do not let DP3 reinstall an older CPU/CUDA PyTorch build over the working RTX 5090 environment.
- Use `pg3d.policies.dp3` for runtime DP3 code. `external/dp3` is kept as a temporary reference
  while the narrow policy/training slice is ported.
