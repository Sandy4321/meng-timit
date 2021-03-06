export KALDI_ROOT=/data/sls/r/u/atitus5/scratch/kaldi 
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh 
# export MENG_ROOT=$PWD 
export MENG_ROOT=/data/sls/r/u/atitus5/meng-timit
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/tools/sph2pipe_v2.5:$MENG_ROOT:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1 
. $KALDI_ROOT/tools/config/common_path.sh 
export LC_ALL=C 

export FEATBIN=$KALDI_ROOT/src/featbin

export SCRATCH=/data/sls/scratch/atitus5/meng
mkdir -p $SCRATCH

export TIMIT_ROOT=/data/sls/static/timit/by_speaker

export MODELS=$SCRATCH/models
mkdir -p $MODELS
export LOGS=$SCRATCH/logs
mkdir -p $LOGS
export CONVOLVED_DATA=$SCRATCH/convolved_data
mkdir -p $CONVOLVED_DATA
export RIRS=$SCRATCH/rirs
mkdir -p $RIRS
export FEATS=$SCRATCH/feats
mkdir -p $FEATS
export RECIPES=$SCRATCH/recipes/timit
mkdir -p $RECIPES

export STEPS=$MENG_ROOT/steps
export UTILS=$MENG_ROOT/utils
export PY_UTILS=$MENG_ROOT/py_utils

# Change to env-cpu if running just on CPU
export TRAIN_ENV=env-gpu     
export AUGMENT_ENV=env-gpu
export HISTOGRAM_ENV=env-gpu
export KALDI_ENV=env-kaldi

export PATH=$PATH:$FEATBIN

# Added so that Anaconda environment activation works
export PATH="/data/sls/r/u/atitus5/scratch/anaconda3/bin:$PATH"

# Added for SOX
export PATH="$PATH:/data/sls/scratch/haotang/opt/bin"
