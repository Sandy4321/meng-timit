#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J augment_vanilla_denoiser

echo "STARTING VANILLA DENOISING MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/base_config.sh

recipe_name=single_rir_vanilla_denoiser_3_16
if [ $# -eq 1 ]; then
    recipe_name=$1
fi

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

augment_log=$LOG_DIR/augment_vanilla_denoiser.log
if [ -f $augment_log ]; then
    # Move old log
    mv $augment_log $LOG_DIR/augment_vanilla_denoiser-$(date +"%F_%T%z").log
fi

export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR

augment_dir=$AUGMENTED_DATA_DIR/vanilla_denoiser
mkdir -p $augment_dir/train
mkdir -p $augment_dir/dev
mkdir -p $augment_dir/test

python3 cnn/scripts/augment_vanilla_denoiser.py &> $augment_log

echo "Undoing CMVN..."
for data_dir in train dev test; do
    echo "Processing $data_dir data"
    for source_class in "${DECODER_CLASSES[@]}"; do
        echo "Processing source $source_class, target clean"
        name=src_${source_class}-tar_clean
        apply-cmvn --norm-vars=true --reverse --utt2spk=ark:$CLEAN_FEATS/${data_dir}/utt2spk scp:$CLEAN_FEATS/${data_dir}/cmvn.scp scp:$augment_dir/${data_dir}/${name}-norm.scp ark,scp:$augment_dir/${data_dir}/${name}.ark,$augment_dir/${data_dir}/${name}.scp
    done
    echo "Done processing $data_dir data"
done

echo "Setting up recipe..."
# dirty_recipe=single_rir_baseline_dirty
dirty_recipe=100_rir_baseline_dirty
./setup_enhancement_recipe.sh $recipe_name $dirty_recipe $augment_dir

echo "DONE VANILLA DENOISING MULTIDECODER DATA AUGMENTATION JOB"
