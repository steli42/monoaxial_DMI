#!/bin/bash
module load lang/Anaconda3/2020.11
# CONDA_HOME=/opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11  # use this on iris
CONDA_HOME=/opt/apps/resif/aion/2020b/epyc/software/Anaconda3/2020.11  # use this on aion
. $CONDA_HOME/etc/profile.d/conda.sh
conda activate kwant_env