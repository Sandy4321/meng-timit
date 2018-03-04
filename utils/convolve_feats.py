#!/usr/bin/env python3

import sys

if len(sys.argv) != 4:
    print("Must specify base WAV SCP file, output data directory and output WAV directory", flush=True)
    sys.exit(1)

base_wav_scp = sys.argv[1]
data_dir = sys.argv[2]
wav_dir = sys.argv[3]

import os
import random
random.seed(1)  # Deterministic for debugging
import subprocess as sp

import numpy as np
import scipy.io.wavfile as wavfile

def pcmToFloat(wav_data):
    return np.divide(wav_data, 2147483647.0).astype(np.float32)

def floatToPCM(wav_data):
    return np.multiply(wav_data, 2147483647.0).astype(np.int32)

# Read in list of RIRs and convert the WAVs to Numpy arrays
print("Reading in RIR paths...", flush=True)
with open(os.path.join(data_dir, "rir.txt"), 'r') as rir_file:
    rir_paths = list(map(lambda x: x.rstrip('\n'), rir_file.readlines()))

print("Converting RIR WAVs to Numpy arrays...", flush=True)
rirs = []
for rir_path in rir_paths:
    # RIRs are already floats -- no need to convert
    fs, rir_data_float = wavfile.read(rir_path)
    rirs.append(rir_data_float)

print("Reading in base WAVs, convolving with RIR WAVs and writing to new WAV SCP...", flush=True)
with open(os.path.join(data_dir, "wav.scp"), 'w') as wav_scp:
    # Check number of lines in base WAV SCP, so we can print progress updates
    with open(base_wav_scp, 'r') as base_wav:
        completed_process = sp.run("wc -l", stdin=base_wav, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True, check=True)
        num_wavs = int(completed_process.stdout)

    update_interval = num_wavs // 20
    with open(base_wav_scp, 'r') as base_wav:
        wavs_processed = 0
        for line in base_wav:
            if "|" in line:
                # Need to run command to get a temporary WAV file
                command = " ".join(line.rstrip('\n').rstrip('|').split(" ")[1:])
                tmp_wav_path = os.path.join(data_dir, "tmp.wav")
                with open(tmp_wav_path, 'w') as tmp_wav:
                    completed_process = sp.run(command, stdout=tmp_wav, stderr=sp.STDOUT, shell=True, check=True)

                # Read in temp WAV file, then delete it
                fs, wav_data_pcm = wavfile.read(tmp_wav_path)
                os.remove(tmp_wav_path)
            else:
                # Read in WAV file
                wav_path = line.rstrip('\n').split(" ")[1]
                fs, wav_data = wavfile.read(wav_path)
            
            # print("Read in %d frames of WAV data (F_s: %d)" % (wav_data_float.shape[0], fs), flush=True)
            
            # Convert to float
            wav_data_float = pcmToFloat(wav_data_pcm)

            # Convolve WAV data with randomly chosen RIR
            rir_idx = random.randint(0, len(rirs) - 1)
            current_rir = rirs[rir_idx]
            convolved_wav_float = np.convolve(wav_data_float, current_rir, mode='same')

            # Convert back to PCM
            convolved_wav_pcm = floatToPCM(convolved_wav_float)

            # Write convolved WAV file
            utt_id = line.split(" ")[0]   # Just use same utterance ID as original for now
            new_wav_path = os.path.join(wav_dir, "%s.wav" % utt_id)
            wavfile.write(new_wav_path, fs, convolved_wav_pcm)

            # Add entry to WAV SCP
            wav_scp.write("%s %s\n" % (utt_id, new_wav_path))

            # Print updates, if any
            wavs_processed += 1
            if wavs_processed % update_interval == 0:
                print("Processed %d/%d lines (%.1f%%)" % (wavs_processed, num_wavs, 100.0 * wavs_processed / num_wavs), flush=True)

print("Done convolving WAVs!", flush=True)
