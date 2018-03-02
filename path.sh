export KALDI_ROOT=/data/sls/r/u/atitus5/scratch/kaldi 
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh 
export MENG_ROOT=$PWD 
export PATH=$MENG_ROOT/utils/:$KALDI_ROOT/tools/openfst/bin:$MENG_ROOT:$PATH 
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1 
. $KALDI_ROOT/tools/config/common_path.sh 
export LC_ALL=C 

export FEATBIN=$KALDI_ROOT/src/featbin

export SCRATCH=/data/sls/scratch/atitus5/meng
mkdir -p $SCRATCH

export MODELS=$SCRATCH/models
mkdir -p $MODELS
export LOGS=$SCRATCH/logs
mkdir -p $LOGS
export AUGMENTED_DATA=$SCRATCH/augmented_data
mkdir -p $AUGMENTED_DATA

export FEATS=$MENG_ROOT
export TEST_FEATS=$MENG_ROOT/test_feats
export UTILS=$MENG_ROOT/utils

# Change to env-cpu if running just on CPU
export TRAIN_ENV=env-gpu     
export AUGMENT_ENV=env-gpu
export PHONES_ENV=env-cpu

export PATH=$PATH:$FEATBIN

# Added so that Anaconda environment activation works
export PATH="/data/sls/r/u/atitus5/scratch/anaconda3/bin:$PATH"

# Added for SOX
export PATH="$PATH:/data/sls/scratch/haotang/opt/bin"
