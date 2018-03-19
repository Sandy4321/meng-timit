export PHONE_FC=( 1024 )
export PHONE_FC_DELIM=$(printf "_%s" "${PHONE_FC[@]}")
export PHONE_ACTIVATION=ReLU

# export CLEAN_RECIPE=${RECIPES}/single_rir_baseline_clean
export CLEAN_RECIPE=${RECIPES}/100_rir_baseline_clean

# export NUM_PHONES=2020
export NUM_PHONES=$(am-info --print-args=false $CLEAN_RECIPE/exp/tri3_ali/final.mdl | grep pdfs | awk '{print $NF}')

export EXPT_NAME="$EXPT_NAME/PHONE_FC_${PHONE_FC_DELIM}_ACT_${PHONE_ACTIVATION}"

export MODEL_DIR=${MODELS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR

export MULTITASK_DIR=${MULTITASK_EXP}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MULTITASK_DIR
