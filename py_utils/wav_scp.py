#!/usr/bin/env python3

import sys

if len(sys.argv) != 3:
    print("Must specify base directory for WAV files, as well as feature directory")
    sys.exit(1)

wav_base_dir = sys.argv[1]
feat_dir = sys.argv[2]

import os

def getWavScpLines(speaker):
    # Why on Earth are there "foo.wav" and "test.wav" in a dataset like this...
    wav_dir = os.path.join(wav_base_dir, speaker)
    wav_filenames = list(filter(lambda x: not (x == "foo.wav" or x == "test.wav" or x == ".tmp.wav"),
                                filter(lambda x: x.endswith(".wav"), os.listdir(wav_dir))))
    wav_scp_lines = []
    for wav_filename in wav_filenames:
        wav_id = wav_filename.split(".")[0]
        utt_id = "%s_%s" % (speaker, wav_id)
        sph2pipe = os.path.join(os.environ["KALDI_ROOT"], "tools/sph2pipe_v2.5/sph2pipe")
        wav_scp_line = "%s %s -f rif %s |\n" % (utt_id, sph2pipe, os.path.join(wav_dir, wav_filename))
        wav_scp_lines.append(wav_scp_line)
    return wav_scp_lines

speakers = os.listdir(wav_base_dir)

print("Combining lines for WAV scp for write...", flush=True)
perdir_lines = map(getWavScpLines, speakers)
all_lines = []
for dir_lines in perdir_lines:
    all_lines.extend(dir_lines)
all_lines.sort()

print("%d lines found. Writing..." % len(all_lines), flush=True)
wav_scp_path = os.path.join(feat_dir, "wav.scp")
with open(wav_scp_path, 'w') as wav_scp:
    for line in all_lines:
        wav_scp.write(line)

print("Successfully wrote files to WAV scp!", flush=True)
