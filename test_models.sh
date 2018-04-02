# Assumes that we've already trained and scored the clean and dirty acoustic models
. ./path.sh
. ./models/base_config.sh

if [ ! -f $MODEL_DIR/best_acoustic_model_clean.pth.tar ] || [ ! -f $MODEL_DIR/best_acoustic_model_dirty.pth.tar ]; then
    echo "Need to train and evaluate baseline acoustic models first"
    exit 1;
else
    if [ ! -d $LOG_DIR/acoustic_model_clean ] || [ ! -d $LOG_DIR/acoustic_model_dirty ]; then
        echo "Already trained acoustic models, but need to evaluate baseline acoustic models first"
        exit 1;
    fi
fi

# Now that we're ready, train ALL the models
model_types=( enhancement_net end2end_phone_net multitask_net enhancement_md multitask_md )
for model_type in "${model_types[@]}"; do
    # Since sbatch now prints "Submitted batch job <X>", need to get last element in
    # space-delimited array to get the actual Job ID
    train_job_output=$(sbatch models/train_${model_type}.bash)
    echo $train_job_output
    train_job_id=$( python3 -c "print(\"$train_job_output\".split(\" \")[-1])" )
    sbatch --dependency=afterok:$train_job_id models/eval_${model_type}.bash
done
