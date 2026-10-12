# Mini pair-representation pretraining ablations

Four sequential runs on one existing 4-node / 16-GPU Ray cluster. Each run uses all 16 GPUs. This folder
contains pretraining only, with no benchmark or downstream evaluation launch.

All runs use `../network/network-mini.yaml`: width 176, 12 PET blocks, 10 local
blocks, and pair width 32 where enabled. Counts below come from constructing the
actual classification model with the shared event schema and synthetic
normalization tensors; they do not include optimizer state or EMA copies.

| YAML | Pair state / classification pair bias | Triangle multiplication | Triangle attention | Contrastive loss | Parameters |
| --- | --- | --- | --- | --- | ---: |
| `no_pair.yaml` | off | off | off | off | 9,055,398 |
| `pair_no_triangle.yaml` | on | off | off | off | 9,713,944 |
| `pair_triangle.yaml` | on | on | on | off | 9,996,568 |
| `pair_triangle_contrastive.yaml` | on | on | on | on | 10,001,848 |

The models share backbone dimensions, not exactly equal parameter counts.
`pair_no_triangle` retains pair embeddings, object-to-pair updates, pair
transitions and pair attention biases, but no triangle operations. Both triangle
runs use identical architecture except the 5,280-parameter contrastive projector.
Classification's direct pair-attention path is enabled consistently in all
three pair variants. The baseline has neither physical pair features nor pair bias.

`use_triangle_multiplication` is a new optional PET setting. Its default is
`true`, preserving existing iterative models. Setting only
`use_triangle_attention: false` in an old config does not disable multiplication.
Synchronize the `evenet` submodule source changes as well as these YAML files.

## Shared controls

- `options.yaml`: 30 execution epochs and 30 schedule epochs, seed 20261013,
  learning rate 0.000012, Lion, weight decay 0.001, EMA, one best checkpoint plus
  `last.ckpt`. The same base classification/noised-classification objectives as
  the earlier pair ablations remain active. Generation, assignment, segmentation
  and regression heads stay disabled.
- Only the fourth run adds truth-supervised pair contrastive loss (scale 0.05,
  temperature 0.1, decay-system/sibling levels, balanced positive/negative groups).
  Its pool is local to each GPU batch; chunk size 32 limits per-chunk work but does
  not cap the total pool or guarantee bounded total backward memory.
- `platform.yaml`: 16 workers / 16 GPUs per run, 20 CPUs per worker, batch 1024
  per GPU, global batch 16384. One trainer requests 16 GPUs and 321 CPUs.
  Run the four variants sequentially. Output paths use the separate
  `mini-1pct-20261013-16gpu-30ep` directory to preserve the earlier campaign.
- `options.yaml` also holds the common normalization path and data selection.
  `dataset_limit: 0.01` applies separately to the training and validation file
  partitions, with the existing `val_split: [0.7, 1.0]` and fixed file-shuffle seed.
  This selects 1% of each partition, not 1% of the whole dataset for training
  plus extra validation. Splitting is by sorted parquet files; limiting is a
  deterministic prefix after seeded file ordering, not uniform row sampling.
  Keep the dataset files and all selection settings fixed across the four runs.
- Validation classification metrics run every epoch. Pair diagnostic plots run
  every 5 epochs in the pair variants. Diffusion is disabled with the generation
  heads. Compare common classification metrics and the 30-epoch checkpoints;
  total `val/loss` includes an additional term in the contrastive run, so its
  absolute value and best-checkpoint epoch are not directly comparable.

Edit `platform.yaml` once to change the batch for all four runs. Start with 1024;
test the triangle+contrastive case before increasing all four to 2048. Triangle
attention scales with object multiplicity, and contrastive candidate comparisons
also grow with batch size. Fewer parameters alone do not ensure that a larger
batch fits. If 1024 fails, lower all four to 512 and use fresh checkpoint paths.

## Launch on the existing cluster

Synchronize the YAML changes, stop any earlier training jobs that occupy this
allocation, and run from the repository root. The checkout and data paths must
be shared across all nodes. Keep `RAY_ADDRESS` pointing to the existing cluster
and export `WANDB_API_KEY`. No launcher, background jobs, or `--load_all` is used.
Each `ray job submit` waits until training finishes; `&&` stops the sequence if
a job fails. The same container Python is used by all entrypoints.

```bash
repo_dir="$PWD"
run_dir="/pscratch/sd/t/tihsu/database/Pretrain-PairRepresentation/mini-1pct-20261013-16gpu-30ep/jobs/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$run_dir"
runtime_env=$(mktemp "$run_dir/.runtime-env.XXXXXX")
python3 - "$runtime_env" <<'PYENV'
import json, os, sys
with open(sys.argv[1], "w") as stream:
    json.dump({"env_vars": {
        "PYTHONPATH": os.getcwd(),
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        "PYTHONUNBUFFERED": "1",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }}, stream)
PYENV

shifter ray job submit --address "$RAY_ADDRESS" --runtime-env "$runtime_env" -- python3 -m scripts.train "$repo_dir/share/pretrain_20261013_pair_mini/no_pair.yaml" --ray_dir "$run_dir/no_pair" &&
shifter ray job submit --address "$RAY_ADDRESS" --runtime-env "$runtime_env" -- python3 -m scripts.train "$repo_dir/share/pretrain_20261013_pair_mini/pair_no_triangle.yaml" --ray_dir "$run_dir/pair_no_triangle" &&
shifter ray job submit --address "$RAY_ADDRESS" --runtime-env "$runtime_env" -- python3 -m scripts.train "$repo_dir/share/pretrain_20261013_pair_mini/pair_triangle.yaml" --ray_dir "$run_dir/pair_triangle" &&
shifter ray job submit --address "$RAY_ADDRESS" --runtime-env "$runtime_env" -- python3 -m scripts.train "$repo_dir/share/pretrain_20261013_pair_mini/pair_triangle_contrastive.yaml" --ray_dir "$run_dir/pair_triangle_contrastive"

rm -f "$runtime_env"
```

The temporary environment file forwards W&B credentials without placing them in
command-line arguments. It uses a shared filesystem path with owner-only file permissions.
Ray Jobs streams logs to the terminal. Its submission output provides the job ID
and commands for checking status and stopping that job. W&B uses project
`Pretrain-Pair-Representation` and entity
`ytchou97-university-of-washington`.

## Timing and next-day inspection

30 epochs is a fixed scientific budget, not a promise of overnight completion.
Measure complete epoch times including validation, use the slowest variant, and
allow extra time for the every-fifth-epoch diagnostic. Initial dataset counting
and loading can make the first epoch atypical. Estimated remaining time is
roughly `(30 - completed_epochs) * representative_epoch_time`.
Check the allocation's remaining wall time (`squeue -j "$SLURM_JOB_ID"`) against
that estimate. A four-hour allocation remains four hours regardless of how many
shells or drivers are running.

Before repeating or resuming, stop the previous Ray job and use a new launcher
log directory. A fresh run needs new checkpoint paths. For a manual native
resume, set `Training.model_checkpoint_load_path` to the actual saved checkpoint
in that run's YAML, keep the same 30-epoch schedule and architecture, and invoke
`python3 -m scripts.train <config.yaml> --ray_dir <new-absolute-ray-directory>`
in the matching environment. Preserve existing checkpoints and use fresh output
paths for a new campaign.

Inspect the saved epoch number, the common classification validation curves,
pair separation and embedding variance/effective rank for the contrastive run.
GPU memory and throughput must be measured on NERSC; local tests cover model
construction, parameter counts, small synthetic forward/backward passes,
triangle compatibility and submission command handling only.
