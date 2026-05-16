#!/bin/bash
#SBATCH --job-name=mcmc_ast
#SBATCH --output=OUTPUT/slurm_%A_%a.log
#SBATCH --error=OUTPUT/slurm_%A_%a.err
#SBATCH --array=0-10                     # single-target CSV: the only valid row index is 0
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --account=bfoc-delta-cpu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

N_CORES=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${N_CORES}
export OPENBLAS_NUM_THREADS=${N_CORES}
export MKL_NUM_THREADS=${N_CORES}
export NUMEXPR_NUM_THREADS=${N_CORES}
export PYTHONUNBUFFERED=1

# MCMC chains = allocated cores.
export NUMPYRO_NUM_CHAINS=${N_CORES}

# ---------------------------------------------------------------------------
# TLS tuning (keep single-threaded to avoid JAX + fork deadlock)
# ---------------------------------------------------------------------------
export TLS_THREADS=1
export TLS_DOWNSAMPLE_STEP=12
export TLS_OVERSAMPLING_FACTOR=1
export TLS_SHOW_PROGRESS=1
export TLS_PRIOR_FRAC_LO=0.9
export TLS_PRIOR_FRAC_HI=1.1
export TLS_MAX_POINTS=1000

# ---------------------------------------------------------------------------
# Compile / cache directories (one per job to avoid collisions)
# ---------------------------------------------------------------------------
export PYTENSOR_FLAGS="base_compiledir=/tmp/pytensor_cache_${SLURM_JOB_ID}"
export ARVIZ_DATA_DIR=/tmp/arviz_${SLURM_JOB_ID}
mkdir -p "$ARVIZ_DATA_DIR"
unset XDG_CACHE_HOME

# ---------------------------------------------------------------------------
# Paths and timestamps
# ---------------------------------------------------------------------------
BASE_DIR="/work/hdd/bfoc/zyan10/TTV_Analysis_For_TESS"
PYTHON_SCRIPT="$BASE_DIR/ppl.py"
MODE="${MODE:-1}"
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
export RUN_TIMESTAMP

# Tidy raw Slurm logs into the matching run directory on exit.  The SBATCH
# directives above must still write to OUTPUT/ first because Slurm opens those
# streams before ppl.py knows the target TIC.
finalize_logs() {
    local raw_log="$BASE_DIR/OUTPUT/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    local raw_err="$BASE_DIR/OUTPUT/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    local fallback_dir="$BASE_DIR/OUTPUT/slurm"
    local dest_dir=""

    shopt -s nullglob
    local candidates=(
        "$BASE_DIR"/OUTPUT/tic_*_index_"${SLURM_ARRAY_TASK_ID}"/mode"${MODE}"/runs/"${RUN_TIMESTAMP}"
    )
    shopt -u nullglob

    if [[ ${#candidates[@]} -gt 0 && -d "${candidates[0]}" ]]; then
        dest_dir="${candidates[0]}/slurm"
    else
        dest_dir="$fallback_dir"
    fi

    mkdir -p "$dest_dir"
    [[ -e "$raw_log" ]] && mv "$raw_log" "$dest_dir/" 2>/dev/null || true
    [[ -e "$raw_err" ]] && mv "$raw_err" "$dest_dir/" 2>/dev/null || true
}
trap finalize_logs EXIT

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID         : $SLURM_JOB_ID"
echo "Array task     : $SLURM_ARRAY_TASK_ID"
echo "Node           : $(hostname)"
echo "CPU cores      : ${N_CORES}  (chains: ${NUMPYRO_NUM_CHAINS})"
echo "Mode           : ${MODE}"
echo "Run timestamp  : ${RUN_TIMESTAMP}"
echo "Start          : $(date)"
echo "=========================================="

cd "$BASE_DIR"
PYTHON_BIN="$BASE_DIR/.venv/bin/python"
if [[ -x "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" -u "$PYTHON_SCRIPT" --index "$SLURM_ARRAY_TASK_ID" --mode "$MODE"
    status=$?
else
    uv run --no-sync python -u "$PYTHON_SCRIPT" --index "$SLURM_ARRAY_TASK_ID" --mode "$MODE"
    status=$?
fi

echo "=========================================="
echo "End            : $(date)"
echo "Exit status    : $status"
echo "=========================================="
exit "$status"
