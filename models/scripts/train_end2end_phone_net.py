import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./models")

from loss_dict import LossDict
from models import End2EndPhoneNet
from py_utils.kaldi_data import KaldiParallelPhoneDataset

# Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py
run_start_t = time.clock()

# Set up feature dimensions
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])

# Set up training parameters
batch_size = int(os.environ["BATCH_SIZE"])
epochs = int(os.environ["EPOCHS"])
optimizer_name = os.environ["OPTIMIZER"]
learning_rate = float(os.environ["LEARNING_RATE"])
on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up the model and associated checkpointing directory
model = End2EndPhoneNet()
print(model, flush=True)
ckpt_path = model.ckpt_path()

# Count number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Model has %d trainable parameters" % params, flush=True)

if on_gpu:
    model.cuda()

# Set up data files
decoder_classes = ["clean"]     # No phone labels available for dirty data

train_scps = []
for decoder_class in decoder_classes:
    train_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "train")
    # train_scp_name = os.path.join(train_scp_dir, "feats.scp")
    train_scp_name = os.path.join(train_scp_dir, "feats-norm.scp")
    train_scps.append(train_scp_name)

dev_scps = []
for decoder_class in decoder_classes:
    dev_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "dev")
    # dev_scp_name = os.path.join(dev_scp_dir, "feats.scp")
    dev_scp_name = os.path.join(dev_scp_dir, "feats-norm.scp")
    dev_scps.append(dev_scp_name)

train_phone_dir = os.path.join(os.environ["DIRTY_FEATS"], "train")
train_phone_scp = os.path.join(train_phone_dir, "phones.scp")

dev_phone_dir = os.path.join(os.environ["DIRTY_FEATS"], "dev")
dev_phone_scp = os.path.join(dev_phone_dir, "phones.scp")

# Set up loss function
nll_criterion = nn.NLLLoss()
if on_gpu:
    nll_criterion = nll_criterion.cuda()
def class_loss(log_probs, target_class):
    return nll_criterion(log_probs, target_class.view((-1)))

# Set up optimizer and associated learning rate scheduler
optimizer = getattr(optim, optimizer_name)(model.parameters(),
                                           lr=learning_rate)
patience = 1
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)

print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up train datasets...", flush=True)
train_dataset = KaldiParallelPhoneDataset(train_scps,
                                          train_phone_scp,
                                         left_context=left_context,
                                         right_context=right_context,
                                         shuffle_utts=True)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          **loader_kwargs)
print("Using %d train features (%d batches)" % (len(train_dataset),
                                                len(train_loader)),
      flush=True)

print("Setting up dev datasets...", flush=True)
dev_dataset = KaldiParallelPhoneDataset(dev_scps,
                                        dev_phone_scp,
                                        left_context=left_context,
                                        right_context=right_context,
                                        shuffle_utts=True)
dev_loader = DataLoader(dev_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        **loader_kwargs)
print("Using %d dev features (%d batches)" % (len(dev_dataset),
                                              len(dev_loader)),
      flush=True)

# Set up logging at a reasonable interval
log_interval = max(1, len(train_loader) // 10)
    

print("Done setting up data.", flush=True)



def train(epoch):
    model.train()

    # Set up some bookkeeping on loss values
    decoder_class_losses = LossDict(decoder_classes, "end2end_phone_net")
    batches_processed = 0

    for batch_idx, (all_feats, all_targets, phones) in enumerate(train_loader):
        # Get data for each decoder class
        feat_dict = dict()
        targets_dict = dict()
        for i in range(len(all_feats)):
            decoder_class = decoder_classes[i]

            feats = all_feats[i]
            targets = all_targets[i]
            if on_gpu:
                feats = feats.cuda()
                targets = targets.cuda()
            feats = Variable(feats)
            targets = Variable(targets)

            feat_dict[decoder_class] = feats
            targets_dict[decoder_class] = targets

        # Set up phones
        if on_gpu:
            phones = phones.cuda()
        phones = Variable(phones)
        
        # Phone classifier training
        optimizer.zero_grad()
        total_loss = 0.0

        feats = feat_dict["clean"]

        # Forward pass through phone classifier
        log_probs = model.classify(feats)

        # Compute class loss and backward pass
        c_loss = class_loss(log_probs, phones)
        total_loss += c_loss

        decoder_class_losses.add("clean", {"phones_xent": c_loss.data[0]})

        # Update generator
        total_loss.backward()
        optimizer.step()

        # Print updates, if any
        batches_processed += 1

        if batches_processed % log_interval == 0:
            print("Train epoch %d: [%d/%d (%.1f%%)]" % (epoch,
                                                        batches_processed,
                                                        len(train_loader),
                                                        batches_processed / len(train_loader) * 100.0),
                  flush=True)
            print(decoder_class_losses, flush=True)
    return decoder_class_losses

def test(epoch, loader):
    model.eval()

    decoder_class_losses = LossDict(decoder_classes, "end2end_phone_net")
    batches_processed = 0

    for batch_idx, (all_feats, all_targets, phones) in enumerate(loader):
        # Get data for each decoder class
        feat_dict = dict()
        targets_dict = dict()
        for i in range(len(all_feats)):
            decoder_class = decoder_classes[i]

            # Set to volatile so history isn't saved (i.e., not training time)
            feats = all_feats[i]
            targets = all_targets[i]
            if on_gpu:
                feats = feats.cuda()
                targets = targets.cuda()
            feats = Variable(feats, volatile=True)
            targets = Variable(targets, volatile=True)

            feat_dict[decoder_class] = feats
            targets_dict[decoder_class] = targets
        
        # Set up phones
        if on_gpu:
            phones = phones.cuda()
        phones = Variable(phones, volatile=True)

        # Phone classifier testing
        feats = feat_dict["clean"]

        # Forward pass through phone classifier
        log_probs = model.classify(feats)

        # Compute class loss
        c_loss = class_loss(log_probs, phones)
        decoder_class_losses.add("clean", {"phones_xent": c_loss.data[0]})
        
        batches_processed += 1

    return decoder_class_losses



# Save model with best dev set loss thus far
best_dev_loss = float('inf')

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

# 1-indexed for pretty printing
print("Starting training!", flush=True)
for epoch in range(1, epochs + 1):
    print("\nSTARTING EPOCH %d" % epoch, flush=True)
    train_start_t = time.clock()

    train_loss_dict = train(epoch)
    print("\nEPOCH %d TRAIN" % epoch, flush=True)
    print(train_loss_dict, flush=True)
            
    dev_loss_dict = test(epoch, dev_loader)
    print("\nEPOCH %d DEV" % epoch, flush=True)
    print(dev_loss_dict, flush=True)
    
    train_end_t = time.clock()
    print("\nEPOCH %d (%.3fs)" % (epoch,
                                  train_end_t - train_start_t),
          flush=True)

    dev_loss = dev_loss_dict.total_loss()
    is_best = (dev_loss <= best_dev_loss)
    if is_best:
        best_dev_loss = dev_loss
        print("\nNew best dev set loss: %.6f" % best_dev_loss, flush=True)

    # Update learning rate scheduler
    scheduler.step(dev_loss)
        
    # Save a checkpoint for our model!
    state_obj = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_dev_loss": best_dev_loss,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_obj, ckpt_path)
    print("Saved checkpoint for model", flush=True)

run_end_t = time.clock()
print("\nCompleted training run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
