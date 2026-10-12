# Mini pair-representation pretraining ablations

Four independent runs on one existing 4-node / 16-GPU Ray cluster. This folder
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

- `options.yaml`: 50 execution epochs and 50 schedule epochs, seed 20261013,
  learning rate 0.000012, Lion, weight decay 0.001, EMA, one best checkpoint plus
  `last.ckpt`. The same base classification/noised-classification objectives as
  the earlier pair ablations remain active. Generation, assignment, segmentation
  and regression heads stay disabled.
- Only the fourth run adds truth-supervised pair contrastive loss (scale 0.05,
  temperature 0.1, decay-system/sibling levels, balanced positive/negative groups).
  Its pool is local to each GPU batch; chunk size 32 limits per-chunk work but does
  not cap the total pool or guarantee bounded total backward memory.
- `platform.yaml`: 4 workers / 4 GPUs per run, 30 CPUs per worker, batch 1024 per
  GPU, global batch 4096. Four trainers request 16 GPUs and 484 CPUs in total,
  leaving some allocation CPUs for data work. Ray handles placement; runs are not
  pinned to particular nodes.
- `options.yaml` also holds the common normalization path and data selection.
  `dataset_limit: 0.01` applies separately to the training and validation file
  partitions, with the existing `val_split: [0.7, 1.0]` and fixed file-shuffle seed.
  This selects 1% of each partition, not 1% of the whole dataset for training
  plus extra validation. Splitting is by sorted parquet files; limiting is a
  deterministic prefix after seeded file ordering, not uniform row sampling.
  Keep the dataset files and all selection settings fixed across the four runs.
- Validation classification metrics run every epoch. Pair diagnostic plots run
  every 5 epochs in the pair variants. Diffusion is disabled with the generation
  heads. Compare common classification metrics and the 50-epoch checkpoints;
  total `val/loss` includes an additional term in the contrastive run, so its
  absolute value and best-checkpoint epoch are not directly comparable.

Edit `platform.yaml` once to change the batch for all four runs. Start with 1024;
test the triangle+contrastive case before increasing all four to 2048. Triangle
attention scales with object multiplicity, and contrastive candidate comparisons
also grow with batch size. Fewer parameters alone do not ensure that a larger
batch fits. If 1024 fails, lower all four to 512 and use fresh checkpoint paths.

## Launch on the existing cluster

Run from this **Pair-Representation EveNet-Full checkout** on a node in your
existing allocation. Keep the same Python/container as the Ray head and workers.
`RAY_ADDRESS` must already point to that cluster and `WANDB_API_KEY` must be set.
The launcher checks data assets, the new triangle switch, empty checkpoint
directories and free cluster resources before launching any training driver.
It does not start or stop Ray. An existing occupied cluster fails the resource
check instead of silently queuing this whole campaign.

For the Shifter cluster used in this conversation:

```bash
shifter --volume=/tmp/ray:/tmp/ray bash \
  share/pretrain_20261013_pair_mini/launch.sh \
  "/pscratch/sd/t/tihsu/database/Pretrain-PairRepresentation/mini-1pct-20261013/jobs/$(date +%Y%m%dT%H%M%S)"
```

If Ray is running directly in your current Python environment:

```bash
bash share/pretrain_20261013_pair_mini/launch.sh \
  "/pscratch/sd/t/tihsu/database/Pretrain-PairRepresentation/mini-1pct-20261013/jobs/$(date +%Y%m%dT%H%M%S)"
```

Use the matching command only once. The shell remains attached while all four
drivers run concurrently. Each has a separate console log and Ray result
directory. A failed run leaves the other runs active and causes a nonzero final
launcher exit. Checkpoints and local logs have separate paths in each run YAML;
W&B uses project `Pretrain-Pair-Representation` and entity
`ytchou97-university-of-washington`. The launcher prints each console-log path.

From another shell, follow a printed log path with `tail -f`. If the first batch
fails, inspect that log before increasing the batch. Confirm that all four runs
report the same train/validation event counts and reach the first checkpoint.

## Timing and next-day inspection

50 epochs is a fixed scientific budget, not a promise of overnight completion.
Measure complete epoch times including validation, use the slowest variant, and
allow extra time for the every-fifth-epoch diagnostic. Initial dataset counting
and loading can make the first epoch atypical. Estimated remaining time is
roughly `(50 - completed_epochs) * representative_epoch_time`.
Check the allocation's remaining wall time (`squeue -j "$SLURM_JOB_ID"`) against
that estimate. A four-hour allocation remains four hours regardless of how many
shells or drivers are running.

Before repeating or resuming, stop the previous driver and use a new launcher
log directory. A fresh run needs new checkpoint paths. For a manual native
resume, set `Training.model_checkpoint_load_path` to the actual saved checkpoint
in that run's YAML, keep the same 50-epoch schedule and architecture, and invoke
`python3 -m scripts.train <config.yaml> --ray_dir <new-absolute-ray-directory>`
in the matching environment. The four-run launcher intentionally refuses
nonempty checkpoint directories and is for fresh campaigns only.

Inspect the saved epoch number, the common classification validation curves,
pair separation and embedding variance/effective rank for the contrastive run.
GPU memory and throughput must be measured on NERSC; local tests cover model
construction, parameter counts, small synthetic forward/backward passes,
triangle compatibility and launcher process handling only.
