export PHONE_FC=( 1024 1024 )
export PHONE_FC_DELIM=$(printf "_%s" "${PHONE_FC[@]}")
export PHONE_ACTIVATION=ReLU

export NUM_PHONES=2095

export EXPT_NAME="$EXPT_NAME/PHONE_FC_${PHONE_FC_DELIM}_ACT_${PHONE_ACTIVATION}"

export MODEL_DIR=${MODELS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR

export MULTITASK_DIR=${MULTITASK_EXP}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MULTITASK_DIR

# Make sure that phones have been generated
for subdir in train dev test; do
    if [ ! -f $DIRTY_FEATS/$subdir/phones.scp ]; then
        echo "Required phone scp not found in $DIRTY_FEATS/$subdir"
        exit 1
    fi
done
