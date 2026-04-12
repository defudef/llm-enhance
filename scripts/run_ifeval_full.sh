#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="artifacts/ifeval-full"
CONTROLLER_CHECKPOINT="artifacts/controller-best.pt"
BASE_ONLY=0
OFFLINE=1
MAX_EXAMPLES=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ifeval_full.sh [options] [-- extra llm-enhance-ifeval args]

Options:
  --output-dir DIR               Where benchmark outputs and logs are written.
                                 Default: artifacts/ifeval-full
  --controller-checkpoint PATH   Controller checkpoint used for the run.
                                 Default: artifacts/controller-best.pt
  --base-only                    Run only the base model benchmark.
  --max-examples N               Optional smoke limit instead of full 541 examples.
  --no-offline                   Allow network access for model resolution.
  -h, --help                     Show this help.

Examples:
  scripts/run_ifeval_full.sh
  scripts/run_ifeval_full.sh --max-examples 10
  scripts/run_ifeval_full.sh --base-only --output-dir artifacts/ifeval-base
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --controller-checkpoint)
      CONTROLLER_CHECKPOINT="$2"
      shift 2
      ;;
    --base-only)
      BASE_ONLY=1
      shift
      ;;
    --max-examples)
      MAX_EXAMPLES="$2"
      shift 2
      ;;
    --no-offline)
      OFFLINE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

RUN_TAG="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="$OUTPUT_DIR/run-$RUN_TAG.log"

CMD=(uv run python -u -m trinity.ifeval)
if [[ "$OFFLINE" -eq 1 ]]; then
  CMD+=(--offline)
fi
CMD+=(--output-dir "$OUTPUT_DIR")
if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$BASE_ONLY" -eq 0 ]]; then
  CMD+=(--controller-checkpoint "$CONTROLLER_CHECKPOINT")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] IFEval run starting"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Root dir: $ROOT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output dir: $OUTPUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: $LOG_FILE"
if [[ "$BASE_ONLY" -eq 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Controller: $CONTROLLER_CHECKPOINT"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Controller: disabled (--base-only)"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: ${CMD[*]}"

set +e
(
  set -o pipefail
  "${CMD[@]}" 2>&1 | while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done | tee -a "$LOG_FILE"
)
CMD_STATUS=$?
set -e

if [[ $CMD_STATUS -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] IFEval run failed with exit code $CMD_STATUS" | tee -a "$LOG_FILE"
  exit $CMD_STATUS
fi

python - <<'PY' "$OUTPUT_DIR" "$BASE_ONLY" | tee -a "$LOG_FILE"
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
base_only = sys.argv[2] == "1"

def print_scores(name: str, path: Path) -> None:
    data = json.loads(path.read_text())
    strict_prompt = data["strict"]["prompt_accuracy"]
    strict_inst = data["strict"]["instruction_accuracy"]
    loose_prompt = data["loose"]["prompt_accuracy"]
    loose_inst = data["loose"]["instruction_accuracy"]
    final = (strict_prompt + strict_inst + loose_prompt + loose_inst) / 4
    print(f"{name}:")
    print(f"  Final Score:        {final:.4f}")
    print(f"  Strict Prompt:      {strict_prompt:.4f}")
    print(f"  Strict Instruction: {strict_inst:.4f}")
    print(f"  Loose Prompt:       {loose_prompt:.4f}")
    print(f"  Loose Instruction:  {loose_inst:.4f}")

base_summary = output_dir / "base" / "summary.json"
if base_summary.exists():
    print_scores("base", base_summary)

controller_summary = output_dir / "controller" / "summary.json"
if not base_only and controller_summary.exists():
    print_scores("controller", controller_summary)
PY

echo "[$(date '+%Y-%m-%d %H:%M:%S')] IFEval run finished successfully" | tee -a "$LOG_FILE"
