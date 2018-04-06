#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --mem=32768
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -J train_multitask_net
#SBATCH --exclude=sls-sm-[5]

echo "STARTING MULTITASK NET HIDDEN VECTOR HISTOGRAM JOB"

. ./path.sh
. ./models/base_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $HISTOGRAM_ENV
echo "Environment set up."

histogram_log=$LOG_DIR/histogram_multitask_net.log
if [ -f $histogram_log ]; then
    # Move old log
    mv $histogram_log $LOG_DIR/histogram_multitask_net-$(date +"%F_%T%z").log
fi

# Compute histogram of euclidean distance between encoded latent vectors for
# parallel clean/dirty pairs
# Stored in .histogram file as "<Distance before enhancement> <Distance after enhancement>"
# for each feature matrix
for subdir in dev test; do
    echo "Computing histograms for $subdir"
    python3 models/scripts/histogram_multitask_net.py $subdir $LOG_DIR/multitask_net_${subdir}.histogram &>> $histogram_log
done

echo "DONE MULTITASK NET HIDDEN VECTOR HISTOGRAM JOB"
