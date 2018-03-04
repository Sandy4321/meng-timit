#!/usr/bin/env python3

import os
import random
import shutil

random.seed(1)  # Fix seed so that we can re-run script

# Pick one RIR randomly from each subdirectory, writing a table of original file location
# to new file location in the process
rir_dest = "/data/sls/scratch/atitus5/meng/rirs"
with open(os.path.join(rir_dest, "locations.txt"), 'w') as locations:
    locations.write("Source\tDestination\n")

    rir_src = "/data/sls/scratch/fgrondin/dataset/ami/rirs/singlemic"
    subdirs = sorted(os.listdir(rir_src))
    for i in range(len(subdirs)):
        subdir = os.path.join(rir_src, subdirs[i])
        rir_wavs = list(filter(lambda filename: filename.endswith(".wav"),
                               os.listdir(subdir)))
        rir_wav_idx = random.randint(0, len(rir_wavs) - 1)

        # Copy RIR to new location with new name (1-indexed version of i)
        rir_wav = os.path.join(subdir, rir_wavs[rir_wav_idx])
        new_rir_wav = os.path.join(rir_dest, "%d.wav" % (i + 1))
        shutil.copy(rir_wav, new_rir_wav)

        # Add table entry
        locations.write("%s\t%s\n" % (rir_wav, new_rir_wav))

