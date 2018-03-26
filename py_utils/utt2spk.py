#!/usr/bin/env python3

import sys

if len(sys.argv) != 3:
    print("Must specify base directory for WAV files, as well as feature directory")
    sys.exit(1)

wav_base_dir = sys.argv[1]
feat_dir = sys.argv[2]

import os

def getUtt2SpkLines(speaker):
    # Why on Earth are there "foo.wav" and "test.wav" in a dataset like this...
    wav_dir = os.path.join(wav_base_dir, speaker)
    wav_filenames = list(filter(lambda x: not (x == "foo.wav" or x == "test.wav" or x == ".tmp.wav"),
                                filter(lambda x: x.endswith(".wav"), os.listdir(wav_dir))))
    utt2spk_lines = []
    for wav_filename in wav_filenames:
        wav_id = wav_filename.split(".")[0]
        utt_id = "%s_%s" % (speaker, wav_id)
        utt2spk_line = "%s %s\n" % (utt_id, speaker)
        utt2spk_lines.append(utt2spk_line)
    return utt2spk_lines

speakers = os.listdir(wav_base_dir)

print("Combining lines for utt2spk for write...", flush=True)
perdir_lines = map(getUtt2SpkLines, speakers)
all_lines = []
for dir_lines in perdir_lines:
    all_lines.extend(dir_lines)
all_lines.sort()

print("%d lines found. Writing..." % len(all_lines), flush=True)
utt2spk_path = os.path.join(feat_dir, "utt2spk")
with open(utt2spk_path, 'w') as utt2spk:
    for line in all_lines:
        utt2spk.write(line)

print("Successfully wrote files to utt2spk!", flush=True)
