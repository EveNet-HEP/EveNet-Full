# Pair-representation quick ablation

This folder reuses the classification-only `pretrain-yulei-20M-3.yaml` setup from
`tmp/EveNet/share/pretrain_20260630_ablation` and changes only the pair path,
run identity, checkpoint directory, and reproducibility controls.

| Config | PairCreator | PET pair integration |
| --- | --- | --- |
| `control.yaml` | disabled | disabled |
| `with_pair_representation.yaml` | `logDeltaR`, `logMass`, `logKT`, `logPTRatio` | `SimpleAddition` attention bias |
| `iterative_update.yaml` | `logDeltaR`, `logMass`, `logKT`, `logPTRatio` | `IterativeUpdate` persistent pair state |
| `hierarchical_contrastive.yaml` | same as iterative | `IterativeUpdate` + training-only truth-bond pair loss |

The baseline runs use the same 20M base network and classification-only objective,
optimizer, learning rate, 30 epochs, random seed, and deterministic parquet-file
order. The hierarchical run matches `iterative_update.yaml` and adds only its
auxiliary loss and projector. Validation loss and classification metrics run
every epoch. Segmentation remains disabled; its truth targets are used only by
the pair diagnostic, while the hierarchical run reads assignment truth for its
auxiliary loss. The current experiment files do not all use the same
`dataset_limit`; set an identical limit and validation partition before treating
their metrics as a matched ablation.

Run from the EveNet-Full repository root:

```bash
python scripts/train.py share/pretrain_20260630_pair_representation/control.yaml
python scripts/train.py share/pretrain_20260630_pair_representation/with_pair_representation.yaml
python scripts/train.py share/pretrain_20260630_pair_representation/iterative_update.yaml
python scripts/train.py share/pretrain_20260630_pair_representation/hierarchical_contrastive.yaml
```

`SimpleAddition` projects the physical pair features directly to PET attention
biases. `IterativeUpdate` embeds the same features into a persistent pair state,
updates it once per PET block, and uses a lightweight `pair_dim: 32` with four
pair heads. Triangle attention remains disabled; triangle multiplication and pair
transitions remain active in every iterative pair-update block. On the first
validation batch of every epoch, the iterative run logs **six pooled scalars
and one overview figure**, regardless of how many processes are present:

- `pair_monitor/separation/all/raw` and `.../pl`: truth-group separation.
- `pair_monitor/separation_gain/all/pl_minus_p0` and `.../pl_minus_raw`:
  differences between separation scores (not scores of difference vectors).
- `pair_monitor/num_pairs` and `.../num_groups`: selected sample coverage.
- `pca/all`: raw, PL, and the vector update PL - P0 in one figure. PCA explained
  variance appears in panel titles, with no separate scalar panels.

Pairs from all processes are pooled before computing scores and PCA; these are
not averages of per-process scores. The existing per-event/per-process truth-group
sampling caps and DDP gather remain in effect, so this is a capped first-batch
diagnostic, not the full validation population. The PCA display-only cap
`Metrics.PairRepresentation.plot_max_pairs_per_group` does not change scalar
scores. Each displayed stage has its own standardized PCA basis.

P0 is the initial pair embedding before PET blocks. It remains the baseline for
PL - P0 but has no standalone plot or scalar by default. No per-process scalar
metrics, separation table, metric bar figures, or process summary are uploaded.
Optional process-name glob patterns add only individual PL - P0 PCA figures:

```yaml
# Under options.Metrics.PairRepresentation
include_all_processes: true
plot_stages: [raw, pl, pl_minus_p0]
process_plots: [] # For example: [WJetsToQQ, "HWW_*"]
```

`include_all_processes` controls the pooled overview and its scalars; it does
not request individual outputs for every process. `plot_stages` selects the
overview panels; an explicit `p0` entry restores that panel if needed.
`process_plots` defaults to an empty list. Each matching process adds one figure
at `pca/<process>` and no extra scalars. Existing W&B runs retain previously
uploaded panels; the smaller output set applies to subsequent logging.

Classification already receives pair information indirectly through PET's
attention bias. A direct, optional head-side path is now available:

```yaml
# Under network
Classification:
  use_pair_representation: true
  pair_attention:
    num_layers: 1
```

The hierarchical configuration enables this path; shared network defaults and
other experiments leave it disabled. It requires PairCreator and
`Body.PET.attention_bias_type: IterativeUpdate`. The final ordered PL is passed
directly into PET's existing `PairToAttentionBias` and `TransformerBlockModule`
implementations, using separate parameters owned by Classification:

```text
PL -> LayerNorm -> linear projection -> bias [B, heads, N, N]
object embeddings -> self-attention(QK^T / sqrt(d) + bias) -> original class-token attention -> logits
```

This follows the pair-information flow in
[Particle Transformer for Jet Tagging, ICML 2022, Section 4, Eqs. (4)-(5)](https://proceedings.mlr.press/v162/qu22b/qu22b.pdf):
pair bias augments particle self-attention; class attention then reads the
updated particles using ordinary attention. Here the input is learned PL and
the blocks are EveNet's existing implementations. This is not a full ParT
reimplementation (its NormFormer blocks and two class-attention blocks are not
copied). One projected PL bias is shared across the optional head-side blocks.

The optional blocks update object embeddings, not PL. Pair directions are kept;
no pair mean pooling or truth labels are used. Padding and diagonal pair entries
have zero bias, while padded attention keys are excluded by the object mask.
An event with no valid off-diagonal pairs bypasses the extra blocks' output.
Classification gradients propagate through the bias into PL. The branch works
without PairContrastive or pair monitoring, in training and inference, and for
both clean and noised classification schedules.

Head attention uses Classification's `num_attention_heads` and `dropout`.
`talking_head`, `layer_scale`, `layer_scale_init`, `drop_probability`, and
`norm_type` inherit `Body.PET` and can be overridden under
`Classification.pair_attention`. PL width follows `Body.PET.pair_dim`, falling
back to PET's hidden dimension when unset.
Setting `use_pair_representation: false` restores the original classification
path and parameter keys. Enabling it adds head parameters, so an old checkpoint
does not contain a trained pair-attention branch; use the existing pretrained
weight-loading path and train the new parameters rather than expecting a strict
resume of the old architecture.

The hierarchical run reuses the existing reco-to-truth assignment indices and
the canonical nested trees in `process_info/pretrain.yaml`. Its YAML block
maps truth-mother aliases such as `W1`, `W2`, `W+`, and `W-` to canonical bond
categories and enables the decay-system and direct-sibling levels. Positives
share a truth mother category within or across local events. Same-event
negatives either share one endpoint with the anchor or are other truth-vetoed
pairs. All local positives and endpoint negatives are kept, while YAML caps
random negatives per anchor. Nothing is gathered across ranks, so normal DDP
gradient synchronization is unchanged.
The trainable projector is a model head; truth labeling and the parameter-free
objective remain in the pair contrastive loss module.
The contrastive head symmetrizes only its projected branch; EveNet's ordered
pair state and inference outputs are unchanged. Set
`Training.Components.PairContrastive.include: false` to disable the branch.

The hierarchical run enables
`options.Training.Components.PairContrastive.candidate_balance: positive_negative`.
For each valid anchor, let `P` contain its positives, `N` contain its endpoint
and sampled random negatives together, and `s = cosine / temperature`. The loss is

```text
L_i = log(mean_{p in P} exp(s_ip) + mean_{n in N} exp(s_in))
      - mean_{p in P} s_ip
```

This adapts the group-count averaging in Zhu et al.,
[Balanced Contrastive Learning, CVPR 2022, Section 3.3, Eq. (6)](https://arxiv.org/html/2207.09052v3#S3.SS3)
to two anchor-relative groups (positive and negative). It is not the full BCL
method: there are no class prototypes, semantic-class averaging, or changes to
anchor/process weights. Endpoint and random negatives remain one pooled group.
The positive target is still averaged over individual positives, preserving
pressure to align all of them rather than only the easiest match.

Each group contributes its mean exponential similarity, removing its raw count
as a multiplicative factor in the denominator. At equal similarities, loss is
`log(2)` and the summed logit gradients of the positive and negative groups are
`-0.5` and `+0.5`, independent of their counts. These are logit derivatives;
they do not guarantee nonzero embedding gradients at exact normalized collapse.
Uniform duplication of either group's candidates preserves the loss and total
gradient; adding genuinely different candidates can still change both.
Stable log-sum-exp arithmetic is used, and anchors lacking either group are
excluded before reductions. Masks, negative sampling, temperature, level weights,
and task loss scale are unchanged. This does not guarantee prevention of collapse.

Set `candidate_balance: none` to restore the original SupCon denominator (a
sum over all candidates). This remains the default for configurations that omit
the option. Raw loss values have a different baseline and gradient scale between
the two modes; compare cosine separation and representation statistics on the
same validation events, not raw loss alone. No extra dashboard panels are added.

Contrastive-specific W&B diagnostics are enabled by
`options.Metrics.PairContrastive.enabled` in `hierarchical_contrastive.yaml`:

The default dashboard contains at most **16 scalars and 2 figures** with both
levels enabled. It keeps the four z cosine means and valid-anchor count per
level, PL/z variance and effective rank, local pair count, and embedding sample
count. Read positive versus negative cosine means together: high positive
similarity alone does not establish separation. The two distribution figures
retain both PL and z, so PL group means do not need separate scalar panels.
Empty comparison groups omit their mean rather than reporting zero.

- `scalar_metrics` selects scalar keys with glob patterns relative to
  `pair_contrastive/`; an empty list disables scalars.
- `cross_process_categories: []` skips process heatmaps and their statistics.
  Use `[W]` to investigate W bonds or `["*"]` for every category.
- `log_cross_process_table: false` suppresses the process comparison table.
  Enabling it includes only categories selected above.

For a temporary full diagnostic run, override these fields under
`options.Metrics.PairContrastive`:

```yaml
scalar_metrics: ["*"]
cross_process_categories: ["*"]
log_cross_process_table: true
```

Available outputs (details require opting in):

- `pair_contrastive/cosine/<level>`: PL and z cosine distributions for
  same-event positives, cross-event positives, endpoint negatives, and sampled
  random negatives. Companion `.../<stage>/<group>_mean` and `_count` scalars
  are selectable; only z means are logged by default. The loss and monitor share truth masks
  and the per-anchor random-negative cap. Histogram anchors must have both a
  positive and a negative, just as in the loss.
- `pair_contrastive/cross_process/<level>/<category>`: mean cosine heatmaps of
  same-category bonds in different events, with comparison counts in each cell.
  All local bonds contribute; missing comparisons show N/A, not zero similarity.
  Diagonal cells compare different events within the same process. Counts are
  directed anchor-to-candidate comparisons. The `cross_process_counts` table
  contains the cell values; `<stage>/mean` and `<stage>/count` scalars cover the
  off-diagonal process comparisons.
- `pair_contrastive/embedding/<stage>/mean_variance` and `effective_rank`:
  collapse diagnostics on unstandardized PL and normalized z. Effective rank
  is the exponential entropy of the centered covariance spectrum; a constant
  embedding has zero variance and rank zero. At least two samples are required.

This output selection does not change loss calculation or the separate
`pair_ssl`, `pair_ssl_val`, and `pair_monitor` namespaces. Existing W&B runs
retain their historical panels; the smaller set applies to new logging and
does not delete previously uploaded charts.

These diagnostics run on rank zero's first validation batch every
`every_n_epochs`, skipping sanity validation, and log at validation epoch end.
They require the contrastive branch to be enabled, reuse its assignment truth,
and work independently of the segmentation-based `Metrics.PairRepresentation`.
They do not gather across ranks: the process coverage describes that local
validation batch, not the entire validation set. The older pair representation
monitor still has its own distributed gather when enabled.

`max_anchors_per_level` limits histogram anchors, `anchor_chunk_size` bounds
each comparison block, and `max_embedding_samples` limits the covariance/SVD
sample. These are display/diagnostic limits only; all events and all loss pairs
are retained. Every sampled histogram anchor sees the complete local pair pool.
The monitor uses a private seeded CPU generator and detached tensors, so it
does not perturb training RNG or gradients. PL uses the same optional pair
symmetrization as the head. No per-coordinate standardization is applied to
the variance or cosine statistics. Use the same validation events and monitor
settings across runs; the sampler seed alone cannot fix changes in batch order.
