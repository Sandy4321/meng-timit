#!/usr/bin/env python3

import sys

if len(sys.argv) != 3:
    print("Must specify data directory and signal-noise ratio", flush=True)
    sys.exit(1)

data_dir = sys.argv[1]
snr = float(sys.argv[2])
assert(snr > 0)

import os
import random

import numpy as np

sys.path.append("./")
from utils.kaldi_data import read_next_utt, write_kaldi_ark, write_kaldi_scp

print("Reading features and writing ark file...", flush=True)
noised_filename_base = "feats-norm"
with open(os.path.join(data_dir, "feats-norm_prenoised.scp"), 'r') as feats_scp:
    with open(os.path.join(data_dir, "%s.ark" % noised_filename_base), 'wb') as noised_feats_ark:
        feats_ark_fd = None
        for line in feats_scp:
            # Read in features first
            utt_id, utt_mat, feats_ark_fd = read_next_utt(line, ark_fd=feats_ark_fd)

            # Generate white noise
            noise_mat = np.random.normal(size=utt_mat.shape) 

            # Add in noise to new matrix and write to ark
            new_utt_mat = utt_mat + ((1.0 / snr) * noise_mat)
            write_kaldi_ark(noised_feats_ark, utt_id, new_utt_mat)

print("Writing scp file...", flush=True)
ark_path = os.path.join(data_dir, "%s.ark" % noised_filename_base)
scp_path = os.path.join(data_dir, "%s.scp" % noised_filename_base)
write_kaldi_scp(ark_path, scp_path)

print("Done noising feats!", flush=True)
