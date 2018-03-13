#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J train_gan

echo "STARTING GAN MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/base_config.sh
. ./cnn/gan_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

train_log=$LOG_DIR/train_gan.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOG_DIR/train_gan-$(date +"%F_%T%z").log
fi

python3 cnn/scripts/train_gan.py &> $train_log

echo "DONE GAN MULTIDECODER TRAINING JOB"
