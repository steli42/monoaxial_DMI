#!/bin/bash
JULIA="julia"
MAIN="tdvp.jl"
CMD="$JULIA $MAIN $1"

echo "Starting simulation... $1"
echo "OMP threads: $OMP_NUM_THREADS"

# source load_conda.sh

$CMD