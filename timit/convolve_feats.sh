#!/bin/bash

# Generate a wav.scp file in the designated recipe's data directory that consists
# of the original TIMIT WAV files convolved with the given room impulse responses (RIRs)

. ./path.sh

if [ $# -ne 1 ]; then
    echo "Must specify recipe name"
    exit 1
fi
recipe_name=$1

recipe_dir=$MENG_ROOT/recipes/$recipe_name
if [ ! -d $recipe_dir ]; then
    echo "Recipe name $recipe_name does not exist in recipes/"
    exit 1
fi

if [ ! -d $recipe_dir/data ]; then
    echo "Data directory not yet created for recipe $recipe_name; please create before running"
    exit 1
fi

for subdir in train dev test; do
    echo "Processing $subdir data"

    current_data=$recipe_dir/data/$subdir
    if [ ! -d $current_data ]; then
        echo "Data subdirectory $subdir not yet created for recipe $recipe_name; please create before running"
        exit 1
    fi

    # Check that we added a RIR list beforehand, so the convolution script knows what to use
    if [ ! -f $current_data/rir.txt ]; then
        echo "RIR file list not provided in subdirectory $subdir; please create before running"
        exit 1
    fi

    # Copy over utt2spk and spk2utt
    cp $FEATS/timit/$subdir/utt2spk $current_data/
    cp $FEATS/timit/$subdir/spk2utt $current_data/

    # Convolve features with given RIRs (provided by recipe)
    echo "Convolving features for subdirectory $subdir"
    mkdir -p $current_data/wavs
    python3 $UTILS/convolve_feats.py $FEATS/timit/$subdir/wav.scp $current_data $current_data/wavs

    echo "Done processing $subdir data."
done

echo "Done data prep for recipe $recipe_name!"
