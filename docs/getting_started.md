# 📘 EveNet Tutorial: From Zero to First Predictions

New to EveNet? This tutorial walks you through the end-to-end workflow so you can move from a clean checkout to model predictions with confidence. Each step links to a deeper reference guide if you need more detail.

---

## 1. EveNet Setup

EveNet requires a controlled environment due to custom dependencies (e.g. `torch-linear-assignment`) and GPU-specific configurations.  
We therefore use a Docker-based workflow together with a local source checkout.

---

### 1. Clone Repository

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
cd EveNet-Full
```

---

### 2. Launch Docker Environment

```bash
docker pull docker.io/avencast1994/evenet:1.5

docker run --gpus all -it \
  -v /path/to/your/data:/workspace/data \
  -v $(pwd):/workspace/EveNet_Full \
  docker.io/avencast1994/evenet:1.5
```

---

### 3. Run EveNet

Inside the container:

```bash
cd /workspace/EveNet_Full

python -m evenet.train share/finetune-example.yaml --ray_dir ~/ray_results
python -m evenet.predict share/predict-example.yaml
```

---

## Notes

- Docker ensures all dependencies (CUDA, PyTorch, and custom libraries) are correctly configured.  
- Running from source allows full flexibility for modifying models, datasets, and training pipelines.  
- This setup is required for reproducing results and is recommended for all users.  

**Setup time:**
- Typical setup time is dominated by downloading the Docker image (~5–10 GB), which usually takes ~5–15 minutes depending on network speed.  
- Container startup and repository cloning typically take less than 1 minute each, resulting in a total setup time of ~10–20 minutes on a standard GPU-enabled desktop.  
- No additional compilation is required, as all dependencies are pre-installed in the Docker image.
---

## 2. Understand the Project Layout (Advanced Users)

Before running any commands, skim these key directories:

- `evenet/` – PyTorch Lightning modules, Ray data pipelines, and trainer utilities.
- `share/` – Ready-to-edit YAML configurations for fine-tuning and prediction.
- `docs/` – Reference documentation that expands on this tutorial. Start with [Model Architecture Tour](model_architecture.md) to see how point-cloud and global features flow through EveNet.

---

## 3. Verify Your Environment

1. **Review cluster helpers as needed.** The `Docker/` and `NERSC/` directories include recipes and SLURM launch scripts tailored for HPC environments.
2. **Confirm GPU visibility (if available).**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

---

## 4. Download Pretrained Weights

EveNet is released as a pretrained foundation model. Start from these weights when fine-tuning or making predictions.

- Browse and download weights directly from HuggingFace:  
  👉 [Avencast/EveNet on HuggingFace](https://huggingface.co/Avencast/EveNet/tree/main)

- Place the downloaded `.ckpt` file somewhere accessible and update your YAML configs with the path (see below).


## 5. Prepare Input Data

Do not run the scripts under `preprocessing/`, which are only for large-scale pretraining. For fine-tuning, prepare your own dataset in the EveNet format.

You are responsible for converting your physics ntuples into the EveNet parquet + metadata format.  
See [data preparation guide](data_preparation.md) for details on schema, normalization, and writer options.  

> 🔍 Tip: Keep your preprocessing configs in version control so you can reproduce and document each dataset.


## 6. Configure an Experiment

> **Note:** The example configs are **not standalone**. Each one uses  
> `default: ...yaml` to load additional base configs. The parser resolves  
> these paths relative to the example’s location, so you must also copy  
> the referenced YAML files and preserve their directory structure.

1. Copy both `share/finetune-example.yaml` (for training) and `share/predict-example.yaml` (for inference) into your working directory.  

2. Update key fields for your experiment:
   - `platform.data_parquet_dir` → directory of your processed parquet files  
   - `Dataset.normalization_file` → path to the normalization statistics you created  
   - `options.Training.pretrain_model_load_path` → pretrained checkpoint to load  
   - `logger` → project name, WANDB API key, or local log directory  

3. For a detailed description of every section and all available overrides, see the [configuration reference](configuration.md).

## 7. Fine-Tune the Model

1. Export your Weights & Biases API key if you plan to log online.
   ```bash
   export WANDB_API_KEY=<your_key>
   ```
2. Launch training with your updated YAML.
   - **Quick start users:** run the packaged CLI after `pip install evenet`.
     ```bash
     evenet-train path/to/your-train-config.yaml
     ```
   - **Source checkout:** execute the module directly to pick up local code edits.
     ```bash
     python -m evenet.train path/to/your-train-config.yaml
     ```
3. Monitor progress:
   - Console output provides per-epoch metrics and checkpoint locations.
   - WANDB dashboards (if enabled) visualize loss curves and system stats.
   - Checkpoints and logs are stored under `options.Training.model_checkpoint_save_path`.


## 8. Generate Predictions

1. Ensure the prediction YAML points to your trained (or pretrained) checkpoint via `options.Training.model_checkpoint_load_path`.
2. Launch inference with either interface.
   ```bash
   # PyPI package
   evenet-predict path/to/your-predict-config.yaml

   # Source checkout
   python -m evenet.predict path/to/your-predict-config.yaml
   ```
3. Outputs land in the configured writers (e.g., parquet, numpy archives). See `docs/predict.md` for writer options and schema notes.


## 9. Explore and Iterate

- Use the artifacts written in the prediction step for downstream analysis (examples live under `downstreams/`).
- Adjust YAML hyperparameters, architecture templates, or preprocessing selections and repeat the workflow.
- When adding new datasets or modules, contribute documentation updates so the next user can follow your path.

Happy exploring! 🚀
