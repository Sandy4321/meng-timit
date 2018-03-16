export FEAT_DIM=40      # 40-dim Mel filter bank
export LEFT_CONTEXT=5
export RIGHT_CONTEXT=5
export OPTIMIZER=Adam
export LEARNING_RATE=0.001
export L2_REG=0.0
export EPOCHS=100
export BATCH_SIZE=64

export ENC_CHANNELS=( 32 32 )
export ENC_KERNELS=( 3 3 )        # Assume square kernels (AxA)
export ENC_DOWNSAMPLES=( 2 2 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling
export ENC_FC=( 256 )     # Fully-connected layers following conv layers

export LATENT_DIM=64

export DEC_FC=( 256 )     # Fully-connected layers before conv layers
export DEC_CHANNELS=( 32 32 )
export DEC_KERNELS=( 3 3 )        # Assume square kernels (AxA)
export DEC_UPSAMPLES=( 2 2 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling

export USE_BATCH_NORM=false
export ACTIVATION_FUNC=ReLU
export WEIGHT_INIT=xavier_uniform

export ENC_CHANNELS_DELIM=$(printf "_%s" "${ENC_CHANNELS[@]}")
export ENC_KERNELS_DELIM=$(printf "_%s" "${ENC_KERNELS[@]}")
export ENC_DOWNSAMPLES_DELIM=$(printf "_%s" "${ENC_DOWNSAMPLES[@]}")
export ENC_FC_DELIM=$(printf "_%s" "${ENC_FC[@]}")

export DEC_FC_DELIM=$(printf "_%s" "${DEC_FC[@]}")
export DEC_CHANNELS_DELIM=$(printf "_%s" "${DEC_CHANNELS[@]}")
export DEC_KERNELS_DELIM=$(printf "_%s" "${DEC_KERNELS[@]}")
export DEC_UPSAMPLES_DELIM=$(printf "_%s" "${DEC_UPSAMPLES[@]}")

export DECODER_CLASSES=( clean dirty )
export DECODER_CLASSES_DELIM=$(printf "_%s" "${DECODER_CLASSES[@]}")

export CLEAN_DATASET=timit_clean
export CLEAN_FEATS=$FEATS/$CLEAN_DATASET
export CLEAN_RECIPE=$RECIPES

export DIRTY_DATASET=timit_dirty_single_rir
# export DIRTY_DATASET=timit_gaussian_5.0
export DIRTY_FEATS=$FEATS/$DIRTY_DATASET


export USE_RECONSTRUCTION=true
export USE_TRANSFORMATION=true

export LOSS_FUNC=L1Loss

export EXPT_NAME="LOSS_${LOSS_FUNC}/RECON_${USE_RECONSTRUCTION}_TRANS_${USE_TRANSFORMATION}_ENC_C${ENC_CHANNELS_DELIM}_K${ENC_KERNELS_DELIM}_P${ENC_DOWNSAMPLES_DELIM}_F${ENC_FC_DELIM}/LATENT_${LATENT_DIM}/DEC_F${DEC_FC_DELIM}_C${DEC_CHANNELS_DELIM}_K${DEC_KERNELS_DELIM}_P${DEC_UPSAMPLES_DELIM}/ACT_${ACTIVATION_FUNC}_BN_${USE_BATCH_NORM}_WEIGHT_INIT_${WEIGHT_INIT}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_L2_REG_${L2_REG}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}"

# For adversarial multidecoders
export DOMAIN_ADV_FC=( 512 512 )
export DOMAIN_ADV_ACTIVATION=Sigmoid
export DOMAIN_ADV_FC_DELIM=$(printf "_%s" "${DOMAIN_ADV_FC[@]}")

# For generative adversarial multidecoders
export GAN_FC=( 512 512 )
export GAN_ACTIVATION=Sigmoid
export GAN_FC_DELIM=$(printf "_%s" "${GAN_FC[@]}")

export MODEL_DIR=${MODELS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $LOG_DIR

# For data augmentation
export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/cnn/$DIRTY_DATASET/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR