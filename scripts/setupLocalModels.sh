#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
HF_PATH="$PROJECT_ROOT/.venv/bin/hf"
FORCE="${1:-}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Create .venv and install requirements.txt before setting up local models." >&2
  exit 1
fi
if [[ ! -x "$HF_PATH" ]]; then
  echo "The setup-only hf CLI is missing. Run: .venv/bin/python -m pip install -r requirements-model-setup.txt" >&2
  exit 1
fi

model_ready() {
  "$PYTHON_PATH" "$PROJECT_ROOT/scripts/verifyLocalModels.py" --model "$1" --files-only --quiet
}

install_model() {
  local name="$1"
  local repository="$2"
  local destination="$3"
  local include="${4:-}"
  if [[ "$FORCE" != "--force" ]] && model_ready "$name"; then
    echo "$name is already complete; skipping."
    return
  fi
  mkdir -p "$PROJECT_ROOT/$destination"
  args=(download "$repository" --local-dir "$PROJECT_ROOT/$destination")
  if [[ -n "$include" ]]; then
    args+=(--include "$include")
  fi
  if [[ "$FORCE" == "--force" ]]; then
    args+=(--force-download)
  fi
  echo "Downloading $name to $PROJECT_ROOT/$destination"
  "$HF_PATH" "${args[@]}"
  model_ready "$name"
}

install_model transcription Systran/faster-whisper-small.en models/faster-whisper-small.en
install_model faceDetection AdamCodd/YOLOv11n-face-detection models/yolo11n-face "*.pt"
install_model semanticAnalysis sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
install_model localCoach Qwen/Qwen2.5-1.5B-Instruct models/qwen2.5-1.5b-instruct

"$PYTHON_PATH" "$PROJECT_ROOT/scripts/verifyLocalModels.py" --files-only
echo "Local model setup complete. Run scripts/verifyLocalModels.py without --files-only to test actual loading."
