#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J augment_vanilla

echo "STARTING VANILLA MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/base_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

augment_log=$LOG_DIR/augment_vanilla.log
if [ -f $augment_log ]; then
    # Move old log
    mv $augment_log $LOG_DIR/augment_vanilla-$(date +"%F_%T%z").log
fi

export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR

augment_dir=$AUGMENTED_DATA_DIR/vanilla
mkdir -p $augment_dir/train
mkdir -p $augment_dir/dev
mkdir -p $augment_dir/test

python3 cnn/scripts/augment_vanilla.py &> $augment_log

echo "Undoing CMVN..."
for data_dir in train dev test; do
    echo "Processing $data_dir data"
    for source_class in "${DECODER_CLASSES[@]}"; do
        for target_class in "${DECODER_CLASSES[@]}"; do
            echo "Processing source $source_class, target $target_class"
            name=src_${source_class}-tar_${target_class}
            apply-cmvn --norm-vars=true --reverse --utt2spk=ark:$CLEAN_FEATS/${data_dir}/utt2spk scp:$CLEAN_FEATS/${data_dir}/cmvn.scp scp:$augment_dir/${data_dir}/${name}-norm.scp ark,scp:$augment_dir/${data_dir}/${name}.ark,$augment_dir/${data_dir}/${name}.scp
        done
    done
    echo "Done processing $data_dir data"
done

echo "DONE VANILLA MULTIDECODER DATA AUGMENTATION JOB"
