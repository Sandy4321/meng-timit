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
    if wav_data.dtype == np.int32 or wav_data.dtype == np.int16:
        if wav_data.dtype == np.int32:
            divisor = 2147483647.0
        else:
            divisor = 32767.0
        return np.divide(wav_data, divisor).astype(np.float32)
    elif wav_data.dtype == np.uint8:
        # Note: unsigned! Need to shift into [0,1] into [-1,1] range
        shifted_wav_data = np.subtract(wav_data, 128, dtype=np.int8)
        divisor = 128.0
        return np.divide(shifted_wav_data, divisor).astype(np.float32)
    else:
        raise TypeError("Unsupported PCM format (%s) -- supported formats are int32, int16 and uint8" % wav_data.dtype)

def floatToPCM(wav_data, dtype=np.int16):
    if dtype == np.int32 or dtype == np.int16:
        if dtype == np.int32:
            factor = 2147483647.0
        else:
            factor = 32767.0
        return np.multiply(wav_data, factor).astype(dtype)
    elif dtype == np.uint8:
        # Note: unsigned! Need to shift into [-1,1] into [0,1] range
        shifted_wav_data = np.add(wav_data, 128, dtype=np.uint8)
        factor = 128.0
        return np.multiply(shifted_wav_data, factor).astype(dtype)
    else:
        raise TypeError("Unsupported PCM format (%s) -- supported formats are int32, int16 and uint8" % wav_data.dtype)

# Read in list of RIRs and convert the WAVs to Numpy arrays
print("Reading in RIR paths...", flush=True)
with open(os.path.join(data_dir, "rir.txt"), 'r') as rir_file:
    rir_paths = list(map(lambda x: x.rstrip('\n'), rir_file.readlines()))

print("Converting RIR WAVs to Numpy arrays...", flush=True)
rirs = []
for rir_path in rir_paths:
    fs, rir_data_pcm = wavfile.read(rir_path)
    rir_data_float = pcmToFloat(rir_data_pcm)
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
            
            # Convert to float
            wav_data_float = pcmToFloat(wav_data_pcm)
            
            # print("Read in %d frames of WAV data (F_s: %d)" % (wav_data_float.shape[0], fs), flush=True)

            # Convolve WAV data with randomly chosen RIR
            rir_idx = random.randint(0, len(rirs) - 1)
            current_rir = rirs[rir_idx]
            convolved_wav_float = np.convolve(wav_data_float, current_rir, mode='same')

            # Handle clipping
            clip_factor = 1.1   # Can be hand-tuned; however, must be strictly > 1
            convolved_wav_float_norm = convolved_wav_float / (max(convolved_wav_float) * clip_factor)

            # Convert back to PCM
            output_type = np.int16
            convolved_wav_pcm = floatToPCM(convolved_wav_float, dtype=output_type)

            '''
            # Sanity check for clipping
            range_info = np.iinfo(output_type)
            assert(max(convolved_wav_pcm) < range_info.max)
            assert(min(convolved_wav_pcm) > range_info.min)
            '''

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
