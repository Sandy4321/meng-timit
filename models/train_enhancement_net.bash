#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J train_enhancement_net
#SBATCH --exclude=sls-sm-[5]

echo "STARTING ENHANCEMENT NET TRAINING JOB"

. ./path.sh
. ./models/base_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

train_log=$LOG_DIR/train_enhancement_net.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOG_DIR/train_enhancement_net-$(date +"%F_%T%z").log
fi

echo "Training model..."
python3 models/scripts/train_enhancement_net.py &> $train_log

echo "DONE ENHANCEMENT NET TRAINING JOB"
