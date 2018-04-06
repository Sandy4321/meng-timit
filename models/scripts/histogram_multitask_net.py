import os
import sys
import time

import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./models")

from models import MultitaskNet
from py_utils.kaldi_data import KaldiParallelDataset

subdir = sys.argv[1]
histogram_file_path = sys.argv[2]

# Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py
run_start_t = time.clock()

# Set up feature dimensions
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])

time_dim = (left_context + right_context + 1)
freq_dim = feat_dim
batch_size = int(os.environ["BATCH_SIZE"])

on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up the model and load checkpoint
model = MultitaskNet()
print(model, flush=True)
ckpt_path = model.ckpt_path()
if on_gpu:
    model.cuda()
ckpt = torch.load(ckpt_path, map_location=lambda storage,loc: storage)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Set up data files
decoder_classes = ["clean", "dirty"]

scps = []
for decoder_class in decoder_classes:
    scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], subdir)
    # scp_name = os.path.join(scp_dir, "feats.scp")
    scp_name = os.path.join(scp_dir, "feats-norm.scp")
    scps.append(scp_name)

print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up dataset...", flush=True)
dataset = KaldiParallelDataset(scps,
                               left_context=left_context,
                               right_context=right_context,
                               shuffle_utts=False)
loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_kwargs)
print("Using %d features (%d batches)" % (len(dataset),
                                          len(loader)),
      flush=True)

# Set up logging at a reasonable interval
log_interval = max(1, len(loader) // 20)
    

print("Done setting up data.", flush=True)

def enhance(model, feats):
    return model.enhance(feats)

def encode(model, feats):
    # Gets latent vector as a 2-D numpy array (batch x latent dim)
    latent_vec, unpool_sizes, pooling_indices = model.encoder(feats.view(-1,
                                                                         1,
                                                                         time_dim,
                                                                         freq_dim))
    return latent_vec.cpu().data.numpy()

def compute_distances(histogram_file_path):
    batches_processed = 0
    with open(histogram_file_path, 'w') as histogram_file:
        for batch_idx, (all_feats, _) in enumerate(loader):
            # Get data for each decoder class
            feat_dict = dict()
            for i in range(len(all_feats)):
                decoder_class = decoder_classes[i]
                feats = all_feats[i]
                if on_gpu:
                    feats = feats.cuda()
                feats = Variable(feats, volatile=True)
                feat_dict[decoder_class] = feats

            clean_feats = feat_dict["clean"]
            dirty_feats = feat_dict["dirty"]

            # Step 1: pre-enhancement
            pre_clean_vec = encode(model, clean_feats)
            pre_dirty_vec = encode(model, dirty_feats)
            
            # Step 2: post-enhancement
            post_clean_vec = encode(model, enhance(model, clean_feats))
            post_dirty_vec = encode(model, enhance(model, dirty_feats))

            # Step 3: compute distances and write to file
            for i in range(pre_clean_vec.shape[0]):
                pre_dist = norm(pre_clean_vec[i, :] - pre_dirty_vec[i, :])
                post_dist = norm(post_clean_vec[i, :] - post_dirty_vec[i, :])
                histogram_file.write("%f %f\n" % (pre_dist, post_dist))

            # Print updates, if any
            batches_processed += 1
            if batches_processed % log_interval == 0:
                print("[%d/%d (%.1f%%)]" % (batches_processed,
                                            len(loader),
                                            batches_processed / len(loader) * 100.0),
                      flush=True)

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

# Actually do it now!
compute_distances(histogram_file_path)

run_end_t = time.clock()
print("\nCompleted histogram run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
