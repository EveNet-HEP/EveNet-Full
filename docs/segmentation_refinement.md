# Segmentation-guided refinement

Refinement is opt-in. Existing configurations instantiate no feedback parameters
and retain their original single-pass behavior and checkpoint keys.

## Configuration

The runnable example is `share/pretrain_20261020_refinement/refinement.yaml`.
It inherits the existing mini experiment's data, normalization, seed, and
resources, enables segmentation supervision, and writes to a separate
`runs/segmentation_refinement` directory relative to the launch directory.
Check the inherited platform/data paths and choose storage appropriate for the
training host before launching through the existing `evenet-train` command.
The example inherits 16 GPU workers and batch size 1024 per worker.

To enable this in another configuration, merge:

```yaml
network:
  Refinement:
    enabled: true
    max_recycles: 2
  Segmentation:
    use_pair_representation: true
options:
  Training:
    Components:
      Segmentation:
        include: true
  Metrics:
    Refinement:
      mask_threshold: 0.5
```

The network must already use `Body.PairCreator` and
`Body.PET.attention_bias_type: IterativeUpdate`. Include a nonzero
`segmentation` loss weight in every relevant progressive-training stage.
The data must contain `segmentation-data` and the class target selected by
`Segmentation.use_full_mask`. Target slots must match
`network.Segmentation.num_queries`; class zero is the existing no-object label.
Dimensions and classes come from the model and event configuration.

`Segmentation.use_pair_representation` defaults to true when refinement is
enabled and false otherwise. It can also be enabled by itself for an ablation.
Its optional `pair_attention` settings follow the classification head's existing
pattern. Feedback parameters belong to PET and use the existing PET optimizer.

## Forward and training semantics

Zero, one, and two recycles mean one, two, and three full reconstruction passes.
Each pass uses the same PET, object encoder, and segmentation weights.
Pair-biased object attention supplies the segmentation decoder with the final
pair state. The next pass starts from the original physical pair embedding plus
a learned projection of

```text
H[b,i,j,c] = sum_q sigmoid(mask[b,q,i]) * sigmoid(mask[b,q,j])
                    * softmax(class[b,q,:])[c], c != null
```

This retains instance identity, is invariant to query-slot permutations, and
preserves null probability without renormalizing the remaining classes.
The first pass has exactly zero feedback. Later all-null predictions can produce
the learned projection bias, distinguishing an empty prediction from no previous
prediction. Padded pairs are masked.

Training samples one depth uniformly from `0..max_recycles`, shared across
distributed workers. Prefix passes run without gradients; only the final pass
contributes supervised losses, including its existing decoder auxiliary losses.
The prediction is detached before feedback projection, while the projection
itself receives gradients on the final pass. Other task heads run on the final
representation. The recycling loop applies to the deterministic schedule.
FAMO's optional post-update loss reevaluation reuses the sampled depth.

Evaluation defaults to `max_recycles`. `EveNetModel.forward` and `shared_step`
accept `num_recycles` for an explicit depth. Evaluation output also contains
`refinement_steps/iteration_N/{pred_logits,pred_masks}` from the same trajectory.
Prediction files retain these tensors for offline comparisons.

## Monitoring

The example recipe explicitly enables `Metrics.PairRepresentation` every epoch.
On the first validation batch from each worker, the existing pair monitor samples
the final refinement pass and logs the pooled `pca/all` figure (raw, PL, and
PL - P0) plus the `pair_monitor/*` separation/count scalars. Per-process figures
remain opt-in through `process_plots`. These diagnostics are separate from the
per-iteration metrics below, which cover the full validation set.

When enabled, every validation batch with an active deterministic schedule
collects every depth, independently of the existing expensive-metrics cadence.
Counts reset at validation start and are summed across workers before division.
The overall scalar keys use:

```text
refinement/train/recycles
refinement/val/iteration_0/...
refinement/val/iteration_1/...
refinement/val/iteration_2/...
```

Each depth reports:

- `dice` and `iou`: soft mask overlap, averaged over matched non-null targets.
- `class_accuracy`: class accuracy on those non-null targets.
- `exact_segment_accuracy`: correct class and exact thresholded membership.
- `exact_event_accuracy`: all target instances reconstructed exactly, with no
  extra non-null prediction in unused slots.
- `corrected_vs_0`, `damaged_vs_0`, `net_corrected_vs_0`: event fractions relative
  to the initial prediction.
- `corrected_vs_previous`, `damaged_vs_previous`: fractions relative to the
  preceding depth.
- `events` and `segments`: denominators for interpreting the rates.

In full logging mode, the same metrics are available under
`refinement/val/process/PROCESS/iteration_N/...` and
`refinement/val/segments/COUNT/iteration_N/...`, where COUNT is the number of
non-null target instances. Hungarian matching reuses the segmentation matcher
with its default costs; padding is removed before matching and scoring.
These diagnostics are unweighted, consistent with the existing segmentation
metrics. Events with no valid input objects are excluded. Overlap and instance
rates are omitted when there are no non-null targets.
The example recipe uses compact logging to prevent one W&B panel per process,
multiplicity, depth, and metric. Configure it under `options.Metrics`:

```yaml
Segmentation:
  wandb_mode: compact
Refinement:
  mask_threshold: 0.5
  wandb_mode: compact
  summary_metrics: [exact_event_accuracy, exact_segment_accuracy, dice, iou, net_corrected_vs_0]
```

Compact refinement logs only the selected overall metrics at each depth (15
scalar curves for three depths) plus one `refinement/val/breakdown` table with
all groups, metrics, and denominators. Missing rates remain missing in the
table. Compact segmentation keeps its two confusion matrices and puts class
and process/multiplicity rates into `segmentation/classes` and
`segmentation/processes` tables. Per-class score histograms are omitted in this
mode. The two monitors therefore emit at most 20 validation panels with this
recipe, independent of the number of processes. The training-depth scalar is
separate. Set either `wandb_mode` to `full` to restore its detailed panels;
omitting the setting preserves the previous full logging behavior.

For an existing W&B workspace, previously created panels remain. Use a manual
workspace and put iteration 0/1/2 on the same line plot for each summary metric,
with epoch as the x-axis. Changing code/YAML requires a new or resumed training
process; it does not update an already-running driver.

When a `LocalLogger` is attached, regardless of W&B mode,
rank zero also appends a long-format `refinement_metrics.csv` in its log directory
with columns `epoch,step,metric,value`; sanity-validation passes are excluded
from this file. This avoids changing the existing training CSV layout.

## Experiment interpretation

Use separate output paths for these controls:

| Run | Segmentation pair input | Refinement enabled |
| --- | --- | --- |
| A | false | false |
| B | true | false |
| C | true | true |

Keep the data, splits, seed, losses, and other heads fixed. A versus B measures
the added pair input; B versus C measures the complete recycling method.
Within C, compare all depths on the same held-out events and checkpoint.
`net_corrected_vs_0` should agree with the change in exact-event accuracy.
Check process and multiplicity breakdowns for degradation hidden by averages.
The monitor does not establish statistical significance or GPU efficiency:
repeat training seeds, use paired event-level uncertainties, and measure
latency/memory on the target hardware before claiming a practical gain.
