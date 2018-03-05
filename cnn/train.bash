#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J train_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
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
        echo "Using generative domain_adversarial net (GAN) style training"
    fi
fi

if [ "$domain_adversarial" == true ]; then
    train_log=$LOG_DIR/train_domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $train_log ]; then
        # Move old log
        mv $train_log $LOG_DIR/train_domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
elif [ "$gan" == true ]; then
    train_log=$LOG_DIR/train_gan_fc_${GAN_FC_DELIM}_act_${GAN_ACT}_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $train_log ]; then
        # Move old log
        mv $train_log $LOG_DIR/train_gan_fc_${GAN_FC_DELIM}_act_${GAN_ACT}_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
else
    train_log=$LOG_DIR/train_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $train_log ]; then
        # Move old log
        mv $train_log $LOG_DIR/train_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
fi

if [ "$PROFILE_RUN" = true ] ; then
    echo "Profiling..."
    python3 cnn/scripts/train_md.py ${run_mode} ${domain_adversarial} ${gan} profile &> $train_log
    echo "Profiling done"
    # echo "Profiling done -- please run 'snakeviz --port=8890 --server $LOG_DIR/train_${run_mode}.prof' to view the results in browser"
else
    python3 cnn/scripts/train_md.py ${run_mode} ${domain_adversarial} ${gan} &> $train_log
fi

echo "DONE MULTIDECODER TRAINING JOB"
