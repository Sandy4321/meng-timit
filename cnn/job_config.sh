export FEAT_DIM=40      # 40 log-Mel
export LEFT_CONTEXT=7
export RIGHT_CONTEXT=7
export OPTIMIZER=Adam
export LEARNING_RATE=0.0001
export EPOCHS=50
export BATCH_SIZE=128

export ENC_CHANNELS=( 256 256 )
export ENC_KERNELS=( 3 3 )        # Assume square kernels (AxA)
export ENC_DOWNSAMPLES=( 3 3 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling
export ENC_FC=( )     # Fully-connected layers following conv layers

export LATENT_DIM=256 

export DEC_FC=( )     # Fully-connected layers before conv layers
export DEC_CHANNELS=( 256 256 )
export DEC_KERNELS=( 3 3 )        # Assume square kernels (AxA)
export DEC_UPSAMPLES=( 3 3 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling

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

export DECODER_CLASSES=( clean reverb )
export DECODER_CLASSES_DELIM=$(printf "_%s" "${DECODER_CLASSES[@]}")

export DATASET_NAME=timit
export CURRENT_FEATS=$FEATS/$DATASET_NAME
export PROFILE_RUN=false

export USE_BACKTRANSLATION=false

export EXPT_NAME="BACKTRANS_${USE_BACKTRANSLATION}_ENC_C${ENC_CHANNELS_DELIM}_K${ENC_KERNELS_DELIM}_P${ENC_DOWNSAMPLES_DELIM}_F${ENC_FC_DELIM}/LATENT_${LATENT_DIM}/DEC_F${DEC_FC_DELIM}_C${DEC_CHANNELS_DELIM}_K${DEC_KERNELS_DELIM}_P${DEC_UPSAMPLES_DELIM}/ACT_${ACTIVATION_FUNC}_BN_${USE_BATCH_NORM}_WEIGHT_INIT_${WEIGHT_INIT}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}"

# For adversarial multidecoders
export DOMAIN_ADV_FC=( 512 512 )
export DOMAIN_ADV_ACTIVATION=LeakyReLU
export DOMAIN_ADV_FC_DELIM=$(printf "_%s" "${DOMAIN_ADV_FC[@]}")

# For generative adversarial multidecoders
export GAN_FC=( 512 512 )
export GAN_ACTIVATION=LeakyReLU
export GAN_FC_DELIM=$(printf "_%s" "${GAN_FC[@]}")

export MODEL_DIR=${MODELS}/cnn/$DATASET_NAME/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/cnn/$DATASET_NAME/$EXPT_NAME
mkdir -p $LOG_DIR

# For data augmentation
export AUGMENTED_DATA_DIR=${AUGMENTED_DATA}/cnn/$DATASET_NAME/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR

# Denoising autoencoder parameters; uses input "destruction" as described in
# "Extracting and Composing Robust Features with Denoising Autoencoders", Vincent et. al.
# http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/176
# Basically sets (NOISE_RATIO * 100)% of input features to 0 at random
export NOISE_RATIO=0.0
