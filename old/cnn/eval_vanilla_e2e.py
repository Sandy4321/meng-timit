import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./cnn")

from setup_md import setup_model, best_ckpt_path
from utils.kaldi_data import KaldiEvalDataset

# Override default pipe handling so that Python doesn't start writing to closed pipe
# and cause a BrokenPipeError
# See https://github.com/python/mypy/issues/2893 for explanation
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

data_dir = sys.argv[1]
source_class = sys.argv[2]

# Set up feature dimensions
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])
time_dim = (left_context + right_context + 1)

# Set up training parameters
on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up the model and associated checkpointing directory
model = setup_model(e2e=True)
if on_gpu:
    model.cuda()
ckpt_path = best_ckpt_path(e2e=True)

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location=lambda storage,loc: storage)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Set up data files
scp_dir = os.path.join(os.environ["%s_FEATS" % source_class.upper()], data_dir)
# scp_name = os.path.join(scp_dir, "feats.scp")
scp_file = os.path.join(scp_dir, "feats-norm.scp")

loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

dataset = KaldiEvalDataset(scp_file,
                           shuffle_utts=False)
loader = DataLoader(dataset,
                    batch_size=1,
                    shuffle=False,
                    **loader_kwargs)

# Print out as Hao format data 
model.eval()
for batch_idx, (feats, _, utt_ids) in enumerate(loader):
    # Only one utterance at once
    utt_id = utt_ids[0]

    # Splice in features
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

    log_probs = model.forward_phones(spliced_feats_tensor)
    print(utt_id, flush=True)
    for i in range(num_frames):
        log_probs_str = " ".join(list(map(str, log_probs.cpu().data.numpy()[i, :].reshape((-1)))))
        print(log_probs_str, flush=True)
    # Print termination character for Hao format
    print(".", flush=True)
