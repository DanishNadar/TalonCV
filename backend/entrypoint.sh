#!/usr/bin/env sh
set -eu
mkdir -p /tmp/taloncv /models
chown taloncv:taloncv /tmp/taloncv /models

if [ -z "${TALONCV_CPU_THREADS:-}" ]; then
  TALONCV_AVAILABLE_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')"
  if [ "$TALONCV_AVAILABLE_CPUS" -gt 1 ]; then
    TALONCV_CPU_THREADS="$((TALONCV_AVAILABLE_CPUS - 1))"
  else
    TALONCV_CPU_THREADS=1
  fi
  export TALONCV_CPU_THREADS
fi
export OMP_NUM_THREADS="$TALONCV_CPU_THREADS"
export MKL_NUM_THREADS="$TALONCV_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$TALONCV_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$TALONCV_CPU_THREADS"
export TOKENIZERS_PARALLELISM=false

exec gosu taloncv python -m backend.entrypoint
