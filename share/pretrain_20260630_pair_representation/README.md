# Pair-representation quick ablation

This folder reuses the classification-only `pretrain-yulei-20M-3.yaml` setup from
`tmp/EveNet/share/pretrain_20260630_ablation` and changes only the pair path,
run identity, checkpoint directory, and reproducibility controls.

| Config | PairCreator | PET pair integration |
| --- | --- | --- |
| `control.yaml` | disabled | disabled |
| `with_pair_representation.yaml` | `logDeltaR`, `logMass`, `logKT`, `logPTRatio` | `SimpleAddition` attention bias |
| `iterative_update.yaml` | `logDeltaR`, `logMass`, `logKT`, `logPTRatio` | `IterativeUpdate` persistent pair state |

All three runs use the same 20M base network, classification-only objective, optimizer,
learning rate, 30 epochs, random seed, and deterministic parquet-file order.
`dataset_limit: 0.001` is applied to both training and validation so validation
does not dominate this quick study. Set `dataset_limit_apply_to_val_set: false`
in all three files if a full, common validation partition is required.

Run from the EveNet-Full repository root:

```bash
python scripts/train.py share/pretrain_20260630_pair_representation/control.yaml
python scripts/train.py share/pretrain_20260630_pair_representation/with_pair_representation.yaml
python scripts/train.py share/pretrain_20260630_pair_representation/iterative_update.yaml
```

`SimpleAddition` projects the physical pair features directly to PET attention
biases. `IterativeUpdate` embeds the same features into a persistent pair state,
updates it once per PET block, and uses Core's defaults of PET hidden dimension
and head count. Triangle attention remains disabled; triangle multiplication
and pair transitions remain active in every iterative pair-update block.
