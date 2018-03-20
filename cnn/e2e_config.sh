export END2END_FC=( 256 256 )
export END2END_FC_DELIM=$(printf "_%s" "${END2END_FC[@]}")
export END2END_ACTIVATION=ReLU

# export CLEAN_RECIPE=${RECIPES}/single_rir_baseline_clean
export CLEAN_RECIPE=${RECIPES}/100_rir_baseline_clean

# export NUM_PHONES=2020
export NUM_PHONES=$(am-info --print-args=false $CLEAN_RECIPE/exp/tri3_ali/final.mdl | grep pdfs | awk '{print $NF}')

export EXPT_NAME="$EXPT_NAME/END2END_FC_${END2END_FC_DELIM}_ACT_${END2END_ACTIVATION}"

export MODEL_DIR=${MODELS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR

export MULTITASK_DIR=${MULTITASK_EXP}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MULTITASK_DIR

# For data augmentation
export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR
