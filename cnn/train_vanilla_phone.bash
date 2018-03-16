#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J train_vanilla_phone

echo "STARTING VANILLA MULTIDECODER MULTITASK TRAINING JOB"

. ./path.sh
. ./cnn/base_config.sh
. ./cnn/phone_config.sh

stage=0
if [ $# -eq 1 ]; then
    stage=$1
fi
echo "Starting from stage $stage"

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

train_log=$LOG_DIR/train_vanilla_phone.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOG_DIR/train_vanilla_phone-$(date +"%F_%T%z").log
fi

if [ $stage -lt 1 ]; then
    # Train model itself
    echo "Training model..."
    python3 cnn/scripts/train_vanilla_phone.py &> $train_log
fi

if [ $stage -lt 2 ]; then
    lang=$AMI/exp/ihm/tri3/graph
    model=$RECIPE/exp/tri3_ali/final.mdl
    for subdir in dev test; do
        echo "Scoring data in $subdir"
        dir=$MULTITASK_DIR/$subdir
        mkdir -p $dir

        # Evaluate model, print log probabilities and generate lattices
        # Use only one job since TIMIT is fairly small
        python3 cnn/scripts/eval_vanilla_phone.py \
            | $UTILS/batch2ark.py \
            | latgen-faster-mapped \
            --max-active=7000 \
            --beam=15 \
            --lattice-beam=7 \
            --acoustic-scale=0.1 \
            --allow-partial=true \
            --word-symbol-table=$lang/words.txt \
            $model \
            $lang/HCLG.fst \
            ark:- \
            "ark:|gzip -c > $dir/lat.1.gz" \
            2> $dir/decode-1.log

        # Score results
        echo "Scoring results for $subdir..."
        num_jobs=1
        echo $num_jobs > $dir/num_jobs
        $STEPS/score.sh $RECIPE/data/$subdir $lang $dir $model >> $score_log 2>&1

        # Print best results
        echo "RESULTS FOR $subdir"
        grep WER $dir/wer_* 2>/dev/null | utils/best_wer.sh
        grep Sum $dir/score_*/*.sys 2>/dev/null | utils/best_wer.sh
    done
fi

echo "DONE VANILLA MULTIDECODER MULTITASK TRAINING JOB"
