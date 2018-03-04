#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 16
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J timit_data_prep

echo "STARTING DATA PREP JOB"

. ./path.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

data_prep_log=$MENG_ROOT/timit/data_prep.log
if [ -f $data_prep_log ]; then
    # Move old log
    mv $data_prep_log $MENG_ROOT/timit/data_prep-$(date +"%F_%T%z").log
fi

echo "Preparing data..."
$MENG_ROOT/timit/timit_data_prep.sh > $data_prep_log
echo "Done preparing data!"

echo "DONE DATA PREP JOB"
