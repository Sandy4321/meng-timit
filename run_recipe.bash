#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 30
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J run_recipe

. ./path.sh

if [ $# -ne 1 ]; then
    echo "Need to specify recipe name"
    exit 1
fi
recipe_name=$1

./train_recipe.bash $recipe_name || exit 1
./eval_recipe.bash $recipe_name || exit 1

echo "Done running recipe $recipe_name"
