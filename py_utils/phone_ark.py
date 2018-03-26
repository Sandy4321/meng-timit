#!/usr/bin/env python3

import sys

if len(sys.argv) != 3:
    print("Must specify base directory for PHN files, as well as feature directory")
    sys.exit(1)

phn_base_dir = sys.argv[1]
feat_dir = sys.argv[2]

from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
import math
import os

import numpy as np

sys.path.append("scripts")
from kaldi_data import write_kaldi_ark
from timit_phone_dict import PHONE_DICT_61, PHONE_DICT_48

SAMPLE_RATE = Decimal(16000)                    # 16 kHz audio
WINDOW_SHIFT = Decimal(10) / Decimal(1000)      # Default Kaldi window for filter bank
WINDOW_SIZE = Decimal(25) / Decimal(1000)       # Default Kaldi window for filter bank

def getPhnLabelVec(phn_filepath):
    label_vec = []
    corrupted = False
    with open(phn_filepath, 'r') as phn_file:
        for line in phn_file:
            if len(line.rstrip('\n').split(" ")) < 3:
                # Rare edge case -- might be TIMIT issue. Skip
                corrupted = True
                break
            start_sample, end_sample, phone = line.rstrip('\n').split(" ")[0:3]
            start_sample = int(start_sample)
            end_sample = int(end_sample)

            '''
            start_frame = int(round((start_sample / float(SAMPLE_RATE)) / WINDOW_SHIFT))
            end_frame = int(round((end_sample / float(SAMPLE_RATE)) / WINDOW_SHIFT))
            '''
            
            # Python round() rounds *.5 up or down somewhat unusually -- use
            # Decimal class to guarantee it is always rounded up
            start_frame = int((Decimal(start_sample) / SAMPLE_RATE/ WINDOW_SHIFT).quantize(0, rounding=ROUND_HALF_UP))
            end_frame = int((Decimal(end_sample) / SAMPLE_RATE/ WINDOW_SHIFT).quantize(0, rounding=ROUND_HALF_UP))

            # For some reason, a few of the utterances don't start at frame 0? What a mess...
            # Add any silence phones at the beginning to compensate
            if len(label_vec) == 0 and start_frame > 0:
                for i in range(start_frame):
                    # label_vec.append(PHONE_DICT_61["h#"])
                    label_vec.append(PHONE_DICT_48["h#"])

            for i in range(start_frame, end_frame):
                # label_vec.append(PHONE_DICT_61[phone])
                label_vec.append(PHONE_DICT_48[phone])

    if corrupted:
        # Give up on this one
        return []

    # Despite all this grief over rounding behavior, we still sometimes doesn't match the
    # frame counts from Kaldi. Worry about this in the data loader instead...
    # Skip the overlap and let scripts/kaldi_data.py handle removing overlap
    '''
    # Handle overlapping windows at end
    snipped_frames = int(((WINDOW_SIZE - WINDOW_SHIFT) / WINDOW_SHIFT).quantize(0, rounding=ROUND_CEILING))
    remaining_frames = len(label_vec) - snipped_frames
    label_vec = label_vec[:remaining_frames]
    '''

    return np.asarray(label_vec, dtype=np.int64).reshape((-1, 1))

def getPhnArkDict(speaker):
    # Why on Earth are there "foo.phn" and "test.phn" in a dataset like this...
    phn_dir = os.path.join(phn_base_dir, speaker)
    phn_filenames = list(filter(lambda x: not (x == "foo.phn" or x == "test.phn" or x == ".tmp.phn"),
                                filter(lambda x: x.endswith(".phn"), os.listdir(phn_dir))))
    phn_dict = dict()
    for phn_filename in phn_filenames:
        phn_id = phn_filename.split(".")[0]
        utt_id = "%s_%s" % (speaker, phn_id)
        label_vec = getPhnLabelVec(os.path.join(phn_dir, phn_filename))
        if len(label_vec) != 0:
            # Filter out short utterances or just garbage
            phn_dict[utt_id] = label_vec
    return phn_dict

speakers = os.listdir(phn_base_dir)

print("Combining dicts for PHN ark for write...", flush=True)
perdir_dicts = map(getPhnArkDict, speakers)
combined_dict = dict()
for dir_dict in perdir_dicts:
    combined_dict = {**combined_dict, **dir_dict}

print("Writing...", flush=True)
phn_ark_path = os.path.join(os.path.join(feat_dir, "data"), "phones.ark")
with open(phn_ark_path, 'wb') as phn_ark:
    for utt_id in sorted(combined_dict.keys()):
        write_kaldi_ark(phn_ark, utt_id, combined_dict[utt_id])

print("Successfully wrote files to PHN ark!", flush=True)
