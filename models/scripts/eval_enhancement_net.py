import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./models")

from models import AcousticModel, EnhancementNet
from py_utils.kaldi_data import KaldiEvalDataset

scp_file = sys.argv[1]

# Override default pipe handling so that Python doesn't start writing to closed pipe
# and cause a BrokenPipeError
# See https://github.com/python/mypy/issues/2893 for explanation
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Set up feature dimensions
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])
time_dim = (left_context + right_context + 1)

on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up baseline model and associated checkpointing directory
acoustic_model = AcousticModel()
am_ckpt_path = acoustic_model.ckpt_path()
if on_gpu:
    acoustic_model.cuda()
am_ckpt = torch.load(am_ckpt_path, map_location=lambda storage,loc: storage)
acoustic_model.load_state_dict(ckpt["state_dict"])
acoustic_model.eval()

# Set up enhancement model and associated checkpointing directory
enhancement_model = EnhancementNet()
en_ckpt_path = enhancement_model.ckpt_path()
if on_gpu:
    enhancement_model.cuda()
en_ckpt = torch.load(en_ckpt_path, map_location=lambda storage,loc: storage)
enhancement_model.load_state_dict(ckpt["state_dict"])
enhancement_model.eval()

# Set up data files
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}
dataset = KaldiEvalDataset(scp_file,
                           shuffle_utts=False)
loader = DataLoader(dataset,
                    batch_size=1,
                    shuffle=False,
                    **loader_kwargs)

# Print out as Hao format data 
for batch_idx, (feats, _, utt_ids) in enumerate(loader):
    # Only one utterance at once
    utt_id = utt_ids[0]

    # Splice in features for input to enhancement model
    feats_numpy = feats.numpy().reshape((-1, feat_dim))
    num_frames = feats_numpy.shape[0]
    spliced_feats = np.empty((num_frames, time_dim, feat_dim))
    for i in range(num_frames):
        frame_spliced = np.zeros((time_dim, feat_dim))
        frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
        frame_spliced[left_context, :] = feats_numpy[i, :]
        frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]

        spliced_feats[i, :, :] = frame_spliced
        
    spliced_feats_tensor = torch.FloatTensor(spliced_feats)
    if on_gpu:
        spliced_feats_tensor = spliced_feats_tensor.cuda()
    spliced_feats_tensor = Variable(spliced_feats_tensor, volatile=True)

    # Enhance features
    en_feats = enhancement_model.forward(spliced_feats_tensor)
    
    # Splice in enhanced features for input to enhancement model
    en_feats_numpy = en_feats.numpy().reshape((-1, feat_dim))
    num_frames = en_feats_numpy.shape[0]
    spliced_en_feats = np.empty((num_frames, time_dim, feat_dim))
    for i in range(num_frames):
        frame_spliced = np.zeros((time_dim, feat_dim))
        frame_spliced[left_context - min(i, left_context):left_context, :] = en_feats_numpy[i - min(i, left_context):i, :]
        frame_spliced[left_context, :] = en_feats_numpy[i, :]
        frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = en_feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]

        spliced_en_feats[i, :, :] = frame_spliced
        
    spliced_en_feats_tensor = torch.FloatTensor(spliced_en_feats)
    if on_gpu:
        spliced_en_feats_tensor = spliced_en_feats_tensor.cuda()
    spliced_en_feats_tensor = Variable(spliced_en_feats_tensor, volatile=True)

    # Compute log probabilities with acoustic model and print to stdout
    log_probs = acoustic_model.classify(spliced_en_feats_tensor)
    print(utt_id, flush=True)
    for i in range(num_frames):
        log_probs_str = " ".join(list(map(str, log_probs.cpu().data.numpy()[i, :].reshape((-1)))))
        print(log_probs_str, flush=True)
    # Print termination character for Hao format
    print(".", flush=True)
