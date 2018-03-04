#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 16
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J convolve_feats

echo "STARTING CONVOLUTION JOB"

. ./path.sh

if [ $# -ne 1 ]; then
    echo "Must specify recipe name"
    exit 1
fi
recipe_name=$1

if [ ! -d recipes/$recipe_name ]; then
    echo "Recipe name $recipe_name does not exist in recipes/"
    exit 1
fi

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

recipe_dir=$LOGS/recipes/$recipe_name
mkdir -p $recipe_dir

data_prep_log=$recipe_dir/convolve_feats.log
if [ -f $data_prep_log ]; then
    # Move old log
    mv $data_prep_log $recipe_dir/convolve_feats-$(date +"%F_%T%z").log
fi

echo "Convolving data..."
$MENG_ROOT/timit/convolve_feats.sh $recipe_name > $data_prep_log
echo "Done convolving data!"

echo "DONE CONVOLUTION JOB"
