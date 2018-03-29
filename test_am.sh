# Assumes that we HAVEN'T already trained and scored the clean and dirty acoustic models
. ./path.sh
. ./models/base_config.sh

if [ -d $LOG_DIR/acoustic_model_clean ] || [ -d $LOG_DIR/acoustic_model_dirty ]; then
    echo "Already evaluated baseline acoustic models"
    exit 1;
else
    if [ -f $MODEL_DIR/best_acoustic_model_clean.pth.tar ] || [ -f $MODEL_DIR/best_acoustic_model_dirty.pth.tar ]; then
        echo "Already trained baseline acoustic models"
        exit 1;
    fi
fi

# Now that we're ready, train the acoustic models
model_types=( clean dirty )
for model_type in "${model_types[@]}"; do
    # Since sbatch now prints "Submitted batch job <X>", need to get last element in
    # space-delimited array to get the actual Job ID
    train_job_output=$( sbatch models/train_acoustic_model.bash $model_type )
    echo $train_job_output
    train_job_id=$( python3 -c "print(\"$train_job_output\".split(\" \")[-1])" )
    sbatch --dependency=afterok:$train_job_id models/eval_acoustic_model.bash $model_type
done
