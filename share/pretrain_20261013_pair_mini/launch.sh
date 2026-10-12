#!/usr/bin/env bash
set -euo pipefail

# Submit to the existing Ray cluster; prefer its dashboard HTTP address.
job_address=${RAY_API_SERVER_ADDRESS:-${RAY_ADDRESS:?Set RAY_API_SERVER_ADDRESS or RAY_ADDRESS}}
: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching}"
max_parallel=4
if [[ "${1:-}" == "-p" || "${1:-}" == "--parallel" ]]; then
    max_parallel=${2:?Missing parallel job count}
    shift 2
fi
if [[ ! "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "Parallel job count must be a positive integer" >&2
    exit 1
fi
run_dir=${1:?Usage: bash launch.sh [-p max_parallel] /absolute/path/to/new/log-directory}
python_bin=${PYTHON_BIN:-python3}
ray_bin=${RAY_BIN:-ray}
config_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$config_dir/../.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
names=(no_pair pair_no_triangle pair_triangle pair_triangle_contrastive)

"$python_bin" - "$config_dir" "${names[@]}" <<'PY'
import inspect
from pathlib import Path
import sys
from evenet.control.global_config import Config
from evenet.network.body.pairformer import PairUpdateBlock

assert 'use_triangle_multiplication' in inspect.signature(PairUpdateBlock).parameters, 'Update the evenet submodule too'
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
    print(f'{name}: {workers} GPUs, batch {config.platform.batch_size} per GPU, {config.options.Training.epochs} epochs')

PY

mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
runtime_env=$(mktemp)
trap 'rm -f "$runtime_env"' EXIT
"$python_bin" - "$repo_dir" "$runtime_env" <<'PYENV'
import json
import os
import sys

# Keep the API key out of command-line arguments and persistent launch logs.
with open(sys.argv[2], 'w') as stream:
    json.dump({'env_vars': {
        'PYTHONPATH': os.environ['PYTHONPATH'],
        'PYTHONUNBUFFERED': '1',
        'WANDB_API_KEY': os.environ['WANDB_API_KEY'],
        # scripts.train also supplies runtime_env to ray.init in Ray 2.44.1.
        'RAY_OVERRIDE_JOB_RUNTIME_ENV': '1',
    }}, stream)
PYENV
pids=()
for name in "${names[@]}"; do
    while (( $(jobs -pr | wc -l) >= max_parallel )); do
        sleep 1
    done
    job_id="pair-mini-$(basename "$run_dir")-$name"
    "$ray_bin" job submit --address "$job_address" \
        --submission-id "$job_id" --runtime-env "$runtime_env" \
        -- python3 -m scripts.train "$config_dir/$name.yaml" \
        --ray_dir "$run_dir/$name/ray" > "$run_dir/$name.log" 2>&1 &
    pids+=("$!")
    printf '%s\t%s\n' "$name" "$job_id" >> "$run_dir/jobs.tsv"
    echo "$name: submitting as $job_id; log: $run_dir/$name.log"
    printf '  ray job logs --address %q %q --follow\n' "$job_address" "$job_id"
    printf '  ray job status --address %q %q\n' "$job_address" "$job_id"
    printf '  ray job stop --address %q %q\n' "$job_address" "$job_id"
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
