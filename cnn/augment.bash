#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J augment_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

if [ "$#" -lt 1 ]; then
    echo "Run mode not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

domain_adversarial=false
gan=false
if [ "$#" -ge 2 ]; then
    if [ "$2" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi
    
    if [ "$2" == "gan" ]; then
        gan=true
        echo "Using generative adversarial net (GAN) style training"
    fi
fi

if [ "$domain_adversarial" == true ]; then
    augment_log=$LOG_DIR/augment_domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOG_DIR/augment_domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}-$(date +"%F_%T%z").log
    fi
    augment_dir=$AUGMENTED_DATA_DIR/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    augment_log=$LOG_DIR/augment_gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOG_DIR/augment_gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
    augment_dir=$AUGMENTED_DATA_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    augment_log=$LOG_DIR/augment_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOG_DIR/augment_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
    augment_dir=$AUGMENTED_DATA_DIR/${run_mode}_ratio${NOISE_RATIO}
fi

mkdir -p $augment_dir

python3 cnn/scripts/augment_md.py ${run_mode} ${domain_adversarial} ${gan} &> $augment_log

echo "Undoing CMVN..."
for data_dir in train dev test; do
    echo "Processing $data_dir data"
    for source_class in "${DECODER_CLASSES[@]}"; do
        for target_class in "${DECODER_CLASSES[@]}"; do
            echo "Processing source $source_class, target $target_class"
            name=${data_dir}-src_${source_class}-tar_${target_class}
            apply-cmvn --norm-vars=true --reverse scp:$CLEAN_FEATS/${data_dir}_cmvn.scp scp:$augment_dir/${name}-norm.scp ark,scp:$augment_dir/${name}.ark,$augment_dir/${name}.scp
        done
    done
    echo "Done processing $data_dir data"
done

echo "DONE MULTIDECODER DATA AUGMENTATION JOB"
