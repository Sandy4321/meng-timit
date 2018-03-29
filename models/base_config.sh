export FEAT_DIM=40      # 40-dim Mel filter bank
export LEFT_CONTEXT=7
export RIGHT_CONTEXT=7
export OPTIMIZER=Adam
export LEARNING_RATE=0.001
export EPOCHS=35
export BATCH_SIZE=128

export CHANNELS=( 64 128 )
export KERNELS=( 5 3 )        # Assume square kernels (AxA)
export DOWNSAMPLES=( 2 0 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling

export LATENT_DIM=1024

export USE_BATCH_NORM=false

export CHANNELS_DELIM=$(printf "_%s" "${CHANNELS[@]}")
export KERNELS_DELIM=$(printf "_%s" "${KERNELS[@]}")
export DOWNSAMPLES_DELIM=$(printf "_%s" "${DOWNSAMPLES[@]}")

export PHONE_FC=( 1024 1024 )
export PHONE_FC_DELIM=$(printf "_%s" "${PHONE_FC[@]}")

export CLEAN_RECIPE=${RECIPES}/100_rir_baseline_clean

# export NUM_PHONES=2020
export NUM_PHONES=$(am-info --print-args=false $CLEAN_RECIPE/exp/tri3_ali/final.mdl | grep pdfs | awk '{print $NF}')

export CLEAN_DATASET=timit_clean
export CLEAN_FEATS=$FEATS/$CLEAN_DATASET

export DIRTY_DATASET=timit_dirty_100_rir
export DIRTY_FEATS=$FEATS/$DIRTY_DATASET

export EXPT_NAME="C${CHANNELS_DELIM}_K${KERNELS_DELIM}_P${DOWNSAMPLES_DELIM}_LATENT_${LATENT_DIM}_PHONE_FC_${PHONE_FC_DELIM}/BN_${USE_BATCH_NORM}_OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}"

export MODEL_DIR=${MODELS}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR
