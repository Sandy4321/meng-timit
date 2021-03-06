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
import subprocess as sp

import numpy as np
import scipy.io.wavfile as wavfile

debug = True
if debug:
    print("Running in debug mode (assertion checks; fixed random seed)", flush=True)
    random.seed(1)  # Deterministic for debugging

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
with open(os.path.join(data_dir, "wav2rir.txt"), 'w') as wav2rir:
    # Write mapping of wav file to the RIR file that was convolved with it for debugging purposes
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
                # Perform zero-padded convolution
                rir_idx = random.randint(0, len(rirs) - 1)
                current_rir = rirs[rir_idx]
                padding = len(current_rir) - 1
                padded_wav_data_float = np.pad(wav_data_float, padding, 'constant', constant_values=0.0)
                convolved_wav_float_full = np.convolve(padded_wav_data_float, current_rir, mode='same')

                # Time-align start of utterance with main peak (i.e. pre-reverb) in RIR, then cut to same sequence length 
                main_peak_idx = np.argmax(current_rir) 
                seq_len = len(wav_data_float)
                start_idx = (padding // 2) + main_peak_idx
                if debug:
                    print("Main peak idx %d (start %d), seq len %d, full len %d" % (main_peak_idx,
                                                                            start_idx,
                                                                            seq_len,
                                                                            len(convolved_wav_float_full)))
                convolved_wav_float = convolved_wav_float_full[start_idx:start_idx + seq_len]
                if debug:
                    print("Final len: %d" % len(convolved_wav_float))
                    assert(len(convolved_wav_float) == seq_len)

                # Normalize based on max volume in original WAV
                max_vol_original = max(abs(wav_data_float))
                max_vol_convolved = max(abs(convolved_wav_float))
                clipping_factor = 0.95      # Tunable; prevents crackling
                norm_factor = clipping_factor * max_vol_original / max_vol_convolved 
                if debug:
                    print("Norm factor: %.3f" % norm_factor)
                convolved_wav_float_norm = convolved_wav_float * norm_factor

                # Convert back to PCM
                output_type = np.int16
                convolved_wav_pcm = floatToPCM(convolved_wav_float_norm, dtype=output_type)

                if debug:
                    # Sanity check for clipping
                    range_info = np.iinfo(output_type)
                    assert(max(convolved_wav_pcm) < range_info.max)
                    assert(min(convolved_wav_pcm) > range_info.min)

                # Write convolved WAV file
                utt_id = line.split(" ")[0]   # Just use same utterance ID as original for now
                new_wav_path = os.path.join(wav_dir, "%s.wav" % utt_id)
                wavfile.write(new_wav_path, fs, convolved_wav_pcm)

                # Add entry to WAV SCP
                wav_scp.write("%s %s\n" % (utt_id, new_wav_path))

                # Add mapping entry to wav2rir
                rir_path = rir_paths[rir_idx]
                wav2rir.write("%s %s\n" % (utt_id, rir_path))

                # Print updates, if any
                wavs_processed += 1
                if wavs_processed % update_interval == 0:
                    print("Processed %d/%d lines (%.1f%%)" % (wavs_processed, num_wavs, 100.0 * wavs_processed / num_wavs), flush=True)

    print("Done convolving WAVs!", flush=True)
