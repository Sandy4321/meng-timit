export GAN_FC=( 512 512 )
export GAN_FC_DELIM=$(printf "_%s" "${GAN_FC[@]}")
export GAN_ACTIVATION=Sigmoid
export GAN_K=1

export EXPT_NAME="$EXPT_NAME/GAN_FC_${GAN_FC_DELIM}_ACT_${GAN_ACTIVATION}_K_${GAN_K}"

export MODEL_DIR=${MODELS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR

# For data augmentation
export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR
