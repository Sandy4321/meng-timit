. ./path.sh
. ./models/base_config.sh

# Train and eval the acoustic models
# Since sbatch now prints "Submitted batch job <X>", need to get last element in
# space-delimited array to get the actual Job ID

# Clean
clean_train_job_output=$( sbatch models/train_acoustic_model.bash clean )
echo $clean_train_job_output
clean_train_job_id=$( python3 -c "print(\"$clean_train_job_output\".split(\" \")[-1])" )
clean_eval_job_output=$( sbatch --dependency=afterany:$clean_train_job_id models/eval_acoustic_model.bash clean )
echo $clean_eval_job_output
clean_eval_job_id=$( python3 -c "print(\"$clean_eval_job_output\".split(\" \")[-1])" )

# Dirty
dirty_train_job_output=$( sbatch models/train_acoustic_model.bash dirty )
echo $dirty_train_job_output
dirty_train_job_id=$( python3 -c "print(\"$dirty_train_job_output\".split(\" \")[-1])" )
sbatch --dependency=afterany:$dirty_train_job_id models/eval_acoustic_model.bash dirty

# Train models that are dependent on the baseline for evaluation
model_types=( enhancement_net enhancement_md )
for model_type in "${model_types[@]}"; do
    # Since sbatch now prints "Submitted batch job <X>", need to get last element in
    # space-delimited array to get the actual Job ID
    train_job_output=$(sbatch models/train_${model_type}.bash)
    echo $train_job_output
    train_job_id=$( python3 -c "print(\"$train_job_output\".split(\" \")[-1])" )
    sbatch --dependency=afterany:$train_job_id models/histogram_${model_type}.bash
    sbatch --dependency=afterany:$train_job_id:$clean_eval_job_id models/eval_${model_type}.bash
done

# Train models that aren't dependent on the baseline
model_types=( end2end_phone_net multitask_net multitask_md )
for model_type in "${model_types[@]}"; do
    # Since sbatch now prints "Submitted batch job <X>", need to get last element in
    # space-delimited array to get the actual Job ID
    train_job_output=$(sbatch models/train_${model_type}.bash)
    echo $train_job_output
    train_job_id=$( python3 -c "print(\"$train_job_output\".split(\" \")[-1])" )
    sbatch --dependency=afterany:$train_job_id models/eval_${model_type}.bash
    
    if [ "$model_type" != "end2end_phone_net" ]; then
        sbatch --dependency=afterany:$train_job_id models/histogram_${model_type}.bash
    fi
done
