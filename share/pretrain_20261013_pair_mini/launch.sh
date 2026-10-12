#!/usr/bin/env bash
set -euo pipefail

# Run inside the same Python/container environment as the existing Ray cluster.
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the existing cluster address}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching}"
run_dir=${1:?Usage: bash launch.sh /absolute/path/to/new/log-directory}
python_bin=${PYTHON_BIN:-python3}
config_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$config_dir/../.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
names=(no_pair pair_no_triangle pair_triangle pair_triangle_contrastive)

"$python_bin" - "$config_dir" "${names[@]}" <<'PY'
import inspect
import os
from pathlib import Path
import sys
import ray
from evenet.control.global_config import Config
from evenet.network.body.pairformer import PairUpdateBlock

assert 'use_triangle_multiplication' in inspect.signature(PairUpdateBlock).parameters, 'Update the evenet submodule too'
required = {'GPU': 0, 'CPU': 0}
for name in sys.argv[2:]:
    config = Config()
    config.load_yaml(Path(sys.argv[1]) / f'{name}.yaml')
    data = config.platform.data_parquet_dir
    assert next(data.glob('*.parquet'), None) is not None, f'No parquet files: {data}'
    assert (data / 'shape_metadata.json').is_file(), f'Missing shape metadata: {data}'
    assert config.options.Dataset.normalization_file.is_file(), 'Missing normalization file'
    checkpoint = config.options.Training.model_checkpoint_save_path
    assert not checkpoint.exists() or not any(checkpoint.iterdir()), f'Use a new checkpoint directory: {checkpoint}'
    workers = config.platform.number_of_workers
    required['GPU'] += workers * config.platform.resources_per_worker.GPU
    required['CPU'] += 1 + workers * config.platform.resources_per_worker.CPU
    print(f'{name}: {workers} GPUs, batch {config.platform.batch_size} per GPU, {config.options.Training.epochs} epochs')
ray.init(address=os.environ['RAY_ADDRESS'])
try:
    available = ray.available_resources()
    for resource, amount in required.items():
        assert available.get(resource, 0) >= amount, f'Need {amount} free {resource}; available: {available}'
    print(f'Existing Ray cluster ready: {required}')
finally:
    ray.shutdown()
PY

mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM
for name in "${names[@]}"; do
    "$python_bin" -m scripts.train "$config_dir/$name.yaml" \
        --ray_dir "$run_dir/$name/ray" > "$run_dir/$name.log" 2>&1 &
    pids+=("$!")
    echo "$name: PID $!, log $run_dir/$name.log"
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "${names[$index]}: completed"
    else
        echo "${names[$index]}: failed; inspect $run_dir/${names[$index]}.log" >&2
        status=1
    fi
done
exit "$status"
