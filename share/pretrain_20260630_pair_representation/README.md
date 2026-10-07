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
validation batch of every epoch, the iterative run logs raw/P0/PL separation
scores, a group-separation table, and PCA explained variance to W&B. Each process
gets PCA distributions at `pca/<process>` and the separation and
signed-gain metrics in a separate `metrics/<process>` figure. The summary
logs per-process separation and gain bar charts at `pair_monitor/summary`. The display-only
sample cap is configured by `Metrics.PairRepresentation.plot_max_pairs_per_group`;
it does not change the scalar metrics.

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
