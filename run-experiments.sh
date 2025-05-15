#!/bin/bash
detect_cores() {
  case "$OSTYPE" in
    linux-gnu* )
      # Native Linux or WSL
      nproc --all
      ;;
    darwin* )
      # macOS
      sysctl -n hw.logicalcpu
      ;;
    msys*|cygwin*|win32 )
      # Native Windows with Git Bash or similar (not WSL)
      WMIC CPU Get NumberOfLogicalProcessors | grep -Eo '[0-9]+' | head -1
      ;;
    *)
      echo "1"  # Fallback default if unknown OS
      ;;
  esac
}
NUM_THREADS=$(detect_cores)
# Optional cap
# MAX_THREADS=8
# NUM_THREADS=$(( NUM_THREADS > MAX_THREADS ? MAX_THREADS : NUM_THREADS ))
echo "Detected $NUM_THREADS logical CPU cores. Will run max($NUM_THREADS - 1, 1)."
NUM_THREADS=$(( NUM_THREADS > 1 ? NUM_THREADS - 1 : 1 ))
#JULIA_NUM_THREADS=$NUM_THREADS julia --project=. experiments-MARX.jl
JULIA_NUM_THREADS=$NUM_THREADS julia --project=. experiments-dmsds.jl
#julia --project=. experiments-dmsds.jl
