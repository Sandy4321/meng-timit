#!/bin/bash

. ./path.sh

timit_feats=$FEATS/timit
echo $timit_feats
mkdir -p $timit_feats

# Create WAV scp
echo "Creating WAV SCP file..."
python3 $UTILS/wav_scp.py $TIMIT_ROOT $timit_feats
echo "Done creating WAV SCP file."

# Create utt2spk
echo "Creating utt2spk..."
python3 $UTILS/utt2spk.py $TIMIT_ROOT $timit_feats
echo "Done creating utt2spk file."

# Create spk2utt
echo "Creating spk2utt..."
$UTILS/utt2spk_to_spk2utt.pl $timit_feats/utt2spk > $timit_feats/spk2utt
echo "Done creating spk2utt file."

# Extract filter bank features
echo "Extracting filter bank features..."
$STEPS/make_fbank.sh --cmd "$UTILS/run.pl" --nj 30 --fbank-config $MENG_ROOT/timit/fbank.conf $timit_feats || exit 1;
echo "Done extracting filter bank features."

# Create frame-level phone labels
# echo "Creating phone labels..."
# python3 $UTILS/phone_ark.py $TIMIT_ROOT $timit_feats
# mv $timit_feats/data/phones.ark $timit_feats/data/phones_tmp.ark
# copy-feats ark:$timit_feats/data/phones_tmp.ark ark,scp:$timit_feats/data/phones.ark,$timit_feats/phones.scp
# rm $timit_feats/data/phones_tmp.ark
# echo "Done creating phone labels."

# Fix any problems that happened
echo "Cleaning up any issues..."
$UTILS/fix_data_dir.sh $timit_feats
echo "Done cleaning up."

# Compute CMVN stats
echo "Computing CMVN stats..."
compute-cmvn-stats scp:$timit_feats/feats.scp ark,scp:$timit_feats/cmvn.ark,$timit_feats/cmvn.scp
echo "Done computing CMVN stats."

# Apply sliding CMVN
echo "Applying sliding CMVN..."
mv $timit_feats/feats.scp $timit_feats/feats_pre_cmvn.scp
norm_vars=false
center=true
cmn_window=300  # Number of frames; 3 seconds 
cmvn_sliding_opts="--norm-vars=$norm_vars --center=$center --cmn-window=$cmn_window"
# apply-cmvn-sliding $cmvn_sliding_opts scp:$timit_feats/feats_pre_cmvn.scp ark,scp:$timit_feats/data/feats_presplice.ark,$timit_feats/feats_presplice.scp
apply-cmvn-sliding $cmvn_sliding_opts scp:$timit_feats/feats_pre_cmvn.scp ark,scp:$timit_feats/data/feats.ark,$timit_feats/feats.scp
echo "Done applying sliding CMVN."

# Splice in context frames
# echo "Splicing in context frames..."
# splice-feats --left-context=$LEFT_SPLICE --right-context=$RIGHT_SPLICE scp:$timit_feats/feats_presplice.scp ark,scp:$timit_feats/data/feats.ark,$timit_feats/feats.scp
# echo "Done splicing in context frames."

# Split into train, validation, evaluation
echo "Subsetting data..."

# 80% train, 10% val, 10% eval
# $UTILS/subset_data_dir_tr_cv.sh --cv-utt-percent 20 $timit_feats $timit_feats/train $timit_feats/val_tmp
# $UTILS/subset_data_dir_tr_cv.sh --cv-utt-percent 50 $timit_feats/val_tmp $timit_feats/val $timit_feats/eval
# rm -r $timit_feats/val_tmp

# Use standard TIMIT splits
$UTILS/subset_data_dir.sh --utt-list $MENG_ROOT/timit/train.flist $timit_feats $timit_feats/train
$UTILS/subset_data_dir.sh --utt-list $MENG_ROOT/timit/dev.flist $timit_feats $timit_feats/dev
$UTILS/subset_data_dir.sh --utt-list $MENG_ROOT/timit/test.flist $timit_feats $timit_feats/test

echo "Done subsetting data."

echo "Done data prep!"
