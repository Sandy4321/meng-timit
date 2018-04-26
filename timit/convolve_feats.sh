#!/bin/bash

# Generate a wav.scp file and associated WAV files that consists of the original
# TIMIT WAV files convolved with the given room impulse responses (RIRs)

. ./path.sh

if [ $# -ne 1 ]; then
    echo "Must specify dataset name"
    exit 1
fi
dataset_name=$1

dataset_dir=$CONVOLVED_DATA/$dataset_name
if [ ! -d $dataset_dir ]; then
    echo "Dataset name $dataset_name does not exist in $CONVOLVED_DATA"
    exit 1
fi

for subdir in train dev test; do
    echo "Processing $subdir data"

    current_data=$dataset_dir/$subdir
    if [ ! -d $current_data ]; then
        echo "Data subdirectory $subdir not yet created for dataset $dataset_name; please create before running"
        exit 1
    fi

    # Check that we added a RIR list beforehand, so the convolution script knows what to use
    if [ ! -f $current_data/rir.txt ]; then
        echo "RIR file list not provided in subdirectory $subdir; please create before running"
        exit 1
    fi

    # Convolve features with given RIRs (provided by dataset)
    echo "Convolving features for subdirectory $subdir"
    mkdir -p $current_data/wavs
    python3 $PY_UTILS/convolve_feats.py recipes/single_rir_baseline_clean/data/$subdir/wav.scp $current_data $current_data/wavs

    echo "Done processing $subdir data."
done

echo "Done data prep for dataset $dataset_name!"
