import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./cnn")

from loss_dict import LossDict
from setup_md import setup_model, best_ckpt_path
from utils.kaldi_data import KaldiEvalDataset, write_kaldi_ark, write_kaldi_scp

run_start_t = time.clock()

# Set up feature dimensions
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])

freq_dim = feat_dim
time_dim = (left_context + right_context + 1)

on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up the model and load its checkpoint
model = setup_model(denoiser=True)
if on_gpu:
    model.cuda()
ckpt_path = best_ckpt_path(denoiser=True)

ckpt = torch.load(ckpt_path, map_location=lambda storage,loc: storage)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Loaded checkpoint; best model ready now", flush=True)

# Set up data files
decoder_classes = []
for res_str in os.environ["DECODER_CLASSES_DELIM"].split("_"):
    if len(res_str) > 0:
        decoder_classes.append(res_str)

train_scps = dict()
for decoder_class in decoder_classes:
    train_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "train")
    # train_scp_name = os.path.join(train_scp_dir, "feats.scp")
    train_scp_name = os.path.join(train_scp_dir, "feats-norm.scp")
    train_scps[decoder_class] = train_scp_name

dev_scps = dict()
for decoder_class in decoder_classes:
    dev_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "dev")
    # dev_scp_name = os.path.join(dev_scp_dir, "feats.scp")
    dev_scp_name = os.path.join(dev_scp_dir, "feats-norm.scp")
    dev_scps[decoder_class] = dev_scp_name

test_scps = dict()
for decoder_class in decoder_classes:
    test_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "test")
    # test_scp_name = os.path.join(test_scp_dir, "feats.scp")
    test_scp_name = os.path.join(test_scp_dir, "feats-norm.scp")
    test_scps[decoder_class] = test_scp_name

print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

# Use batch size of 1, so we just see one utterance at a time
train_loaders = dict()
dev_loaders = dict()
test_loaders = dict()
for decoder_class in decoder_classes:
    print("=> Setting up %s..." % decoder_class, flush=True)

    print("===> Setting up train dataset...", flush=True)
    train_dataset = KaldiEvalDataset(train_scps[decoder_class],
                                     shuffle_utts=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              **loader_kwargs)
    train_loaders[decoder_class] = train_loader
    print("===> Using %d train features (%d batches)" % (len(train_dataset),
                                                    len(train_loader)),
          flush=True)
    
    print("===> Setting up dev dataset...", flush=True)
    dev_dataset = KaldiEvalDataset(dev_scps[decoder_class],
                                   shuffle_utts=False)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=1,
                            shuffle=False,
                            **loader_kwargs)
    dev_loaders[decoder_class] = dev_loader
    print("===> Using %d dev features (%d batches)" % (len(dev_dataset),
                                                  len(dev_loader)),
          flush=True)
    
    print("===> Setting up test dataset...", flush=True)
    test_dataset = KaldiEvalDataset(test_scps[decoder_class],
                                    shuffle_utts=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             **loader_kwargs)
    test_loaders[decoder_class] = test_loader
    print("===> Using %d test features (%d batches)" % (len(test_dataset),
                                                   len(test_loader)),
          flush=True)

output_dir = os.path.join(os.environ["AUGMENTED_DATA_DIR"], "vanilla_denoiser")
    
print("Done setting up data.", flush=True)



def augment(loader, augmented_dir, source_class, target_class):
    # Set up logging at a reasonable interval
    log_interval = max(1, len(loader) // 10)

    # Set up some bookkeeping
    batches_processed = 0

    # Write ark file first
    ark_path = os.path.join(augmented_dir, "src_%s-tar_%s-norm.ark" % (source_class, target_class))
    with open(ark_path, 'wb') as ark_fd:
        for batch_idx, (feats, _, utt_id_arr) in enumerate(loader):
            utt_id = utt_id_arr[0]  # Only one utterance

            # Run batch through target decoder
            feats_numpy = feats.numpy().reshape((-1, freq_dim))
            num_frames = feats_numpy.shape[0]
            frame_spliced_all = np.empty((num_frames, time_dim, freq_dim))
            for i in range(num_frames):
                frame_spliced = np.zeros((time_dim, freq_dim))
                frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
                frame_spliced[left_context, :] = feats_numpy[i, :]
                frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]
                
                frame_spliced_all[i, :, :] = frame_spliced
                
            # Pass full utterance through decoder
            frame_tensor = torch.FloatTensor(frame_spliced_all)
            if on_gpu:
                frame_tensor = frame_tensor.cuda()
            frame_tensor = Variable(frame_tensor)

            recon_frames = model.forward_decoder(frame_tensor, target_class)

            # Write to output file
            decoded_feats = recon_frames.cpu().data.numpy().reshape((-1, time_dim, freq_dim))[:, left_context:left_context + 1, :].reshape((-1, freq_dim))
            write_kaldi_ark(ark_fd, utt_id, decoded_feats)

            # Print updates, if any
            batches_processed += 1
            if batches_processed % log_interval == 0:
                print("Augmented [%d/%d (%.1f%%)]" % (batches_processed,
                                                      len(loader),
                                                      batches_processed / len(loader) * 100.0),
                      flush=True)

    # Write corresponding SCP file
    scp_path = os.path.join(augmented_dir, "src_%s-tar_%s-norm.scp" % (source_class, target_class))
    write_kaldi_scp(ark_path, scp_path)

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

print("Starting data augmentation!", flush=True)

print("Processing train data", flush=True)
augment_start_t = time.clock()
augmented_dir = os.path.join(output_dir, "train")
for source_class in decoder_classes:
    print("Processing source %s, target clean" % source_class, flush=True)
    augment_start_t = time.clock()

    augment(train_loaders[source_class], augmented_dir, source_class, "clean")
    
    augment_end_t = time.clock()
    print("\nProcessed source %s, target clean in %.3fs" % (source_class,
                                                            augment_end_t - augment_start_t),
          flush=True)
augment_end_t = time.clock()
print("Processed train data in %.3fs" % (augment_end_t - augment_start_t), flush=True)

print("Processing dev data", flush=True)
augment_start_t = time.clock()
augmented_dir = os.path.join(output_dir, "dev")
for source_class in decoder_classes:
    print("Processing source %s, target clean" % source_class, flush=True)
    augment_start_t = time.clock()

    augment(dev_loaders[source_class], augmented_dir, source_class, "clean")
    
    augment_end_t = time.clock()
    print("\nProcessed source %s, target clean in %.3fs" % (source_class,
                                                            augment_end_t - augment_start_t),
          flush=True)
augment_end_t = time.clock()
print("Processed dev data in %.3fs" % (augment_end_t - augment_start_t), flush=True)

print("Processing test data", flush=True)
augment_start_t = time.clock()
augmented_dir = os.path.join(output_dir, "test")
for source_class in decoder_classes:
    print("Processing source %s, target clean" % source_class, flush=True)
    augment_start_t = time.clock()

    augment(test_loaders[source_class], augmented_dir, source_class, "clean")
    
    augment_end_t = time.clock()
    print("\nProcessed source %s, target clean in %.3fs" % (source_class,
                                                            augment_end_t - augment_start_t),
          flush=True)
augment_end_t = time.clock()
print("Processed test data in %.3fs" % (augment_end_t - augment_start_t), flush=True)


run_end_t = time.clock()
print("\nCompleted data augmentation run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
