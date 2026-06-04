#!/bin/bash
# Submit the formula-AS final notebook pipeline as Snellius jobs.
#
# Required environment:
#   PROJECT_DIR=/home/<user>/thesis
#   DATA_DIR=/scratch-shared/<user>/datasets
#   CONDA_ENV=mysimenv

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-mysimenv}"
MAX_CONCURRENT="${MAX_CONCURRENT:-6}"
B2_ALPHAS="${B2_ALPHAS:-0.1 1.0 10.0}"
RUN_NB5="${RUN_NB5:-0}"
RUN_NB6="${RUN_NB6:-1}"
RUN_SHAP="${RUN_SHAP:-0}"
SMOKE="${SMOKE:-0}"

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR is required, e.g. export DATA_DIR=/scratch-shared/<user>/datasets" >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs results/eval_parts results/job_metadata

RUNNER="scripts/snellius/run_python_job.sh"

submit_job() {
  local name="$1"
  local script="$2"
  local args="${3:-}"
  local dep="${4:-}"
  local array="${5:-}"
  local time_limit="${6:-06:00:00}"

  local cmd=(sbatch --parsable --job-name "$name" --time "$time_limit")
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency "$dep")
  fi
  if [[ -n "$array" ]]; then
    cmd+=(--array "$array")
  fi
  cmd+=(--export=ALL,PROJECT_DIR="$PROJECT_DIR",DATA_DIR="$DATA_DIR",CONDA_ENV="$CONDA_ENV",SCRIPT_PATH="$script",SCRIPT_ARGS="$args",TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}",ABLATION_TIMESTEPS="${ABLATION_TIMESTEPS:-}",B2_ALPHAS="$B2_ALPHAS",EVALUATION_ROLLOUTS="${EVALUATION_ROLLOUTS:-}",CVAR_N_WINDOWS="${CVAR_N_WINDOWS:-}",CVAR_ALPHA="${CVAR_ALPHA:-}",CVAR_TIGHTEN="${CVAR_TIGHTEN:-}",FORMULA_GAMMA_MIN="${FORMULA_GAMMA_MIN:-}",FORMULA_GAMMA_MAX="${FORMULA_GAMMA_MAX:-}",FORMULA_SKEW_TICKS_MAX="${FORMULA_SKEW_TICKS_MAX:-}")
  cmd+=("$RUNNER")
  "${cmd[@]}"
}

if [[ "$SMOKE" == "1" ]]; then
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000}"
  export ABLATION_TIMESTEPS="${ABLATION_TIMESTEPS:-10000}"
  export CVAR_N_WINDOWS="${CVAR_N_WINDOWS:-5}"
  TEST_ARRAY="0-1%2"
  AGG_PARTIAL_ARG="--allow-partial"
else
  TEST_ARRAY="0-22%${MAX_CONCURRENT}"
  AGG_PARTIAL_ARG=""
fi

echo "Submitting formula-AS pipeline from $PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "B2_ALPHAS=$B2_ALPHAS"
echo "TEST_ARRAY=$TEST_ARRAY"

prepare_job=$(submit_job "prep_formula" "scripts/snellius/prepare_formula_pipeline.py" "--force" "" "" "01:00:00")
b0_job=$(submit_job "b0_fair" "scripts/snellius/run_b0_fair.py" "" "afterok:${prepare_job}" "" "04:00:00")

b1_train=$(submit_job "b1_train" "scripts/snellius/train_formula_b1.py" "" "afterok:${prepare_job}" "" "12:00:00")
b1_eval=$(submit_job "b1_eval" "scripts/snellius/evaluate_formula_model.py" "--model b1" "afterok:${b1_train}" "$TEST_ARRAY" "03:00:00")
b1_agg=$(submit_job "b1_agg" "scripts/snellius/aggregate_final_results.py" "--model b1 ${AGG_PARTIAL_ARG}" "afterok:${b1_eval}" "" "01:00:00")

b2_train_jobs=()
b2_eval_jobs=()
b2_agg_jobs=()
for alpha in $B2_ALPHAS; do
  train_job=$(submit_job "b2_a${alpha}" "scripts/snellius/train_formula_b2.py" "--alpha ${alpha}" "afterok:${prepare_job}" "" "12:00:00")
  eval_job=$(submit_job "b2e_a${alpha}" "scripts/snellius/evaluate_formula_model.py" "--model b2 --alpha ${alpha}" "afterok:${train_job}" "$TEST_ARRAY" "03:00:00")
  agg_job=$(submit_job "b2g_a${alpha}" "scripts/snellius/aggregate_final_results.py" "--model b2 --alpha ${alpha} ${AGG_PARTIAL_ARG}" "afterok:${eval_job}" "" "01:00:00")
  b2_train_jobs+=("$train_job")
  b2_eval_jobs+=("$eval_job")
  b2_agg_jobs+=("$agg_job")
done

b2_agg_dep=$(IFS=:; echo "afterok:${b2_agg_jobs[*]}")
b2_select=$(submit_job "b2_select" "scripts/snellius/aggregate_final_results.py" "--select-b2" "$b2_agg_dep" "" "01:00:00")

cvar_cal=$(submit_job "b3_cvarcal" "scripts/snellius/calibrate_cvar_b3.py" "" "afterok:${b1_train}" "" "03:00:00")
b3_train=$(submit_job "b3_train" "scripts/snellius/train_formula_b3.py" "" "afterok:${cvar_cal}:${b2_select}" "" "12:00:00")
b3_eval=$(submit_job "b3_eval" "scripts/snellius/evaluate_formula_model.py" "--model b3" "afterok:${b3_train}" "$TEST_ARRAY" "03:00:00")
b3_agg=$(submit_job "b3_agg" "scripts/snellius/aggregate_final_results.py" "--model b3 ${AGG_PARTIAL_ARG}" "afterok:${b3_eval}" "" "01:00:00")

if [[ "$SMOKE" == "1" ]]; then
  comparison="skipped_smoke_mode"
else
  compare_dep="afterok:${b0_job}:${b1_agg}:${b2_select}:${b3_agg}"
  comparison=$(submit_job "final_compare" "scripts/snellius/aggregate_final_results.py" "--comparison" "$compare_dep" "" "01:00:00")
fi

if [[ "$RUN_NB6" == "1" ]]; then
  ablation_jobs=()
  for feature_set in base_state base_plus_vol full; do
    ablation_jobs+=("$(submit_job "abl_${feature_set}" "scripts/snellius/run_feature_ablation.py" "--feature-set ${feature_set}" "afterok:${prepare_job}" "" "04:00:00")")
  done
  ablation_dep=$(IFS=:; echo "afterok:${ablation_jobs[*]}")
  submit_job "abl_agg" "scripts/snellius/aggregate_final_results.py" "--ablation" "$ablation_dep" "" "01:00:00" >/dev/null
fi

if [[ "$RUN_SHAP" == "1" ]]; then
  submit_job "b1_shap" "scripts/snellius/run_shap_diagnostics.py" "" "afterok:${b1_train}" "" "04:00:00" >/dev/null
fi

echo "Submitted jobs:"
echo "  prepare      $prepare_job"
echo "  B0           $b0_job"
echo "  B1 train     $b1_train"
echo "  B1 eval      $b1_eval"
echo "  B2 select    $b2_select"
echo "  B3 train     $b3_train"
echo "  comparison   $comparison"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName,State,ExitCode,Elapsed -X"
