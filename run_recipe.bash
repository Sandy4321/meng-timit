#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 30
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J run_recipe
#SBATCH --exclude=sls-sm-5

. ./path.sh

if [ $# -ne 1 ]; then
    echo "Need to specify recipe name"
    exit 1
fi
recipe_name=$1

echo "Using recipe name $recipe_name"
if [ ! -d recipes/$recipe_name ]; then
    echo "Recipe $recipe_name does not exist in recipes/"
    exit 1
fi

recipe_log_dir=$LOGS/recipes
recipe_log=$recipe_log_dir/${recipe_name}.log
if [ -f $recipe_log ]; then
    # Move old log
    mv $recipe_log $recipe_log_dir/${recipe_name}-$(date +"%F_%T%z").log
fi

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

cd recipes/$recipe_name
./run.sh > $recipe_log

echo "Done running recipe $recipe_name"
