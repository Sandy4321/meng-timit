#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 30
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J build_kaldi_cuda
#SBATCH --exclude=sls-sm-5

. ./path.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

cd $KALDI_ROOT/src
./configure --shared --use-cuda
make depend -j 30 
make -j 30 

echo "Done building Kaldi with CUDA support"
