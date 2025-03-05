#!/bin/bash
JULIA="julia --heap-size-hint=180G"
MAIN="find_state.jl"
CMD="$JULIA $MAIN $1"

echo "Starting simulation... $1"
echo "OMP threads: $OMP_NUM_THREADS"

$CMD