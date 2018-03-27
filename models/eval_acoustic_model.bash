#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 16
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J eval_acoustic_model

echo "STARTING ACOUSTIC MODEL EVAL JOB"

. ./path.sh
. ./models/base_config.sh
. ./models/acoustic_model_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

lang=$CLEAN_RECIPE/exp/tri3/graph
model=$CLEAN_RECIPE/exp/tri3_ali/final.mdl

eval_log=$LOG_DIR/eval_acoustic_model.log
if [ -f $eval_log ]; then
    # Move old log
    mv $eval_log $LOG_DIR/eval_acoustic_model-$(date +"%F_%T%z").log
fi

# Decode and evaluate the acoustic model
for source_class in clean dirty; do
    echo "Processing source class $source_class"
    if [ "$source_class" == "clean" ]; then
        feats=$CLEAN_FEATS
    else
        feats=$DIRTY_FEATS
    fi

    for subdir in dev test; do
        echo "Scoring data in $subdir"
        
        dir=$LOG_DIR/$subdir/$source_class
        mkdir -p $dir
        
        num_jobs=10
        echo $num_jobs > $dir/num_jobs

        if [ ! -f $dir/lat.1.gz ]; then
            # Split feats scp
            split_scps=""
            for n in $(seq $num_jobs); do
                split_scps="$split_scps $feats/$subdir/feats-norm.$n.scp"
            done
            $UTILS/split_scp.pl $feats/$subdir/feats-norm.scp $split_scps || exit 1;

            # Evaluate model, print log probabilities and generate lattices
            echo "Generating lattices..."
            $UTILS/run.pl JOB=1:$num_jobs $dir/decode-JOB.log \
                python3 models/scripts/eval_acoustic_model.py $feats/$subdir/feats-norm.JOB.scp \
                    \| $PY_UTILS/batch2ark.py \
                    \| latgen-faster-mapped \
                    --max-active=7000 \
                    --beam=15 \
                    --lattice-beam=7 \
                    --acoustic-scale=0.1 \
                    --allow-partial=true \
                    --word-symbol-table=$lang/words.txt \
                    $model \
                    $lang/HCLG.fst \
                    ark:- \
                    "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1
        else
            echo "Lattice already generated; skipping"
        fi

        # Score results
        echo "Scoring $subdir results for $source_class..."
        local/score.sh $CLEAN_RECIPE/data/$subdir $lang $dir $model >> $eval_log 2>&1

        # Print best results
        echo "RESULTS FOR $subdir, class $source_class"
        grep WER $dir/wer_* 2>/dev/null | $UTILS/best_wer.sh
        grep Sum $dir/score_*/*.sys 2>/dev/null | $UTILS/best_wer.sh
    done
done

echo "DONE ACOUSTIC MODEL EVAL JOB"
