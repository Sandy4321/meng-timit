#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J run_acoustic_model

echo "STARTING ACOUSTIC MODEL JOB"

. ./path.sh
. ./models/base_config.sh
. ./models/acoustic_model_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

run_log=$LOG_DIR/run.log
if [ -f $run_log ]; then
    # Move old log
    mv $run_log $LOG_DIR/run-$(date +"%F_%T%z").log
fi

# First, train the acoustic model
python3 models/scripts/train_acoustic_model.py &> $run_log

echo "DONE ACOUSTIC MODEL JOB"
