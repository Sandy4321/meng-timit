#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_enhancement_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING ENHANCEMENT MULTIDECODER TRAINING JOB"

. ./path.sh
. ./models/base_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

train_log=$LOG_DIR/train_enhancement_md.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOG_DIR/train_enhancement_md-$(date +"%F_%T%z").log
fi

echo "Training model..."
python3 models/scripts/train_enhancement_md.py &> $train_log

echo "DONE ENHANCEMENT MULTIDECODER TRAINING JOB"
