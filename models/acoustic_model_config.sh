export PHONE_FC=( 1024 1024 1024 )
export PHONE_FC_DELIM=$(printf "_%s" "${PHONE_FC[@]}")

export ACOUSTIC_MODEL_DECODER_CLASSES=( clean )
export ACOUSTIC_MODEL_DECODER_CLASSES_DELIM=$(printf "_%s" "${ACOUSTIC_MODEL_DECODER_CLASSES[@]}")

# export CLEAN_RECIPE=${RECIPES}/single_rir_baseline_clean
export CLEAN_RECIPE=${RECIPES}/100_rir_baseline_clean

# export NUM_PHONES=2020
export NUM_PHONES=$(am-info --print-args=false $CLEAN_RECIPE/exp/tri3_ali/final.mdl | grep pdfs | awk '{print $NF}')

export EXPT_NAME="$EXPT_NAME/PHONE_FC_${PHONE_FC_DELIM}"

export MODEL_DIR=${MODELS}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR
