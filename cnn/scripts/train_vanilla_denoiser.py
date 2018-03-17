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
from utils.kaldi_data import KaldiParallelDataset

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
l2_reg = float(os.environ["L2_REG"])
use_reconstruction = True if os.environ["USE_RECONSTRUCTION"] == "true" else False
use_transformation = True if os.environ["USE_TRANSFORMATION"] == "true" else False
loss_func = os.environ["LOSS_FUNC"]
on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed_all(1)

# Set up the model and associated checkpointing directory
model = setup_model(denoiser=True)
print(model, flush=True)

# Count number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Model has %d trainable parameters" % params, flush=True)

if on_gpu:
    model.cuda()
ckpt_path = best_ckpt_path(denoiser=True)

# Set up data files
decoder_classes = []
for res_str in os.environ["DECODER_CLASSES_DELIM"].split("_"):
    if len(res_str) > 0:
        decoder_classes.append(res_str)

training_scps = []
for decoder_class in decoder_classes:
    training_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "train")
    # training_scp_name = os.path.join(training_scp_dir, "feats.scp")
    training_scp_name = os.path.join(training_scp_dir, "feats-norm.scp")
    training_scps.append(training_scp_name)

dev_scps = []
for decoder_class in decoder_classes:
    dev_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "dev")
    # dev_scp_name = os.path.join(dev_scp_dir, "feats.scp")
    dev_scp_name = os.path.join(dev_scp_dir, "feats-norm.scp")
    dev_scps.append(dev_scp_name)

# Set up loss function
criterion = getattr(nn, loss_func)() 
if on_gpu:
    criterion.cuda()
def reconstruction_loss(recon_x, x):
    return criterion(recon_x, x)

# Set up optimizer for the generator portion (multidecoder itself)
generator_optimizer = getattr(optim, optimizer_name)(model.generator_parameters(),
                                                     lr=learning_rate,
                                                     weight_decay=l2_reg)

print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up training datasets...", flush=True)
training_dataset = KaldiParallelDataset(training_scps,
                                        left_context=left_context,
                                        right_context=right_context,
                                        shuffle_utts=True)
training_loader = DataLoader(training_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             **loader_kwargs)
print("Using %d training features (%d batches)" % (len(training_dataset),
                                                   len(training_loader)),
      flush=True)

print("Setting up dev datasets...", flush=True)
dev_dataset = KaldiParallelDataset(dev_scps,
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
log_interval = max(1, len(training_loader) // 10)
    

print("Done setting up data.", flush=True)



def train(epoch):
    model.train()

    # Set up some bookkeeping on loss values
    decoder_class_losses = LossDict(decoder_classes, denoiser=True)
    batches_processed = 0

    for batch_idx, (all_feats, all_targets) in enumerate(training_loader):
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
        

        # STEP 1: Generator training


        if use_reconstruction:
            # PHASE 1: Backprop clean data through same decoder (reconstruction)
            generator_optimizer.zero_grad()

            feats = feat_dict["clean"]
            targets = targets_dict["clean"]

            # Forward pass through same decoder
            recon_batch = model.forward_decoder(feats, "clean")

            # Compute reconstruction loss and backward pass
            r_loss = reconstruction_loss(recon_batch, targets)
            r_loss.backward()

            decoder_class_losses.add("clean", {"autoencoding_recon_loss": r_loss.data[0]})

            # Update generator
            generator_optimizer.step()
        

        if use_transformation:
            # PHASE 2: Backprop dirty data through clean decoder (denoising/transformation)
            generator_optimizer.zero_grad()

            feats = feat_dict["dirty"]
            targets_other = targets_dict["clean"]

            # Forward pass through other decoder
            transformed_feats = model.forward_decoder(feats, "clean")

            # Compute reconstruction loss and backward pass
            r_loss = reconstruction_loss(transformed_feats, targets_other)
            r_loss.backward()
            
            decoder_class_losses.add("dirty", {"transformation_recon_loss": r_loss.data[0]})
            
            # Update generator
            generator_optimizer.step()

        # Print updates, if any
        batches_processed += 1

        if batches_processed % log_interval == 0:
            print("Train epoch %d: [%d/%d (%.1f%%)]" % (epoch,
                                                        batches_processed,
                                                        len(training_loader),
                                                        batches_processed / len(training_loader) * 100.0),
                  flush=True)
            print(decoder_class_losses, flush=True)
    return decoder_class_losses

def test(epoch, loader):
    model.eval()

    decoder_class_losses = LossDict(decoder_classes, denoiser=True)
    batches_processed = 0

    for batch_idx, (all_feats, all_targets) in enumerate(loader):
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


        # STEP 1: Autoencoder testing


        if use_reconstruction:
            # PHASE 1: Backprop clean data through clean decoder (reconstruction)
            feats = feat_dict["clean"]
            targets = targets_dict["clean"]

            # Run features through same decoder
            recon_batch = model.forward_decoder(feats, "clean")
            r_loss = reconstruction_loss(recon_batch, targets)

            decoder_class_losses.add("clean", {"autoencoding_recon_loss": r_loss.data[0]})
        
        if use_transformation:
            # PHASE 2: Backprop dirty data through clean decoder (denoising/transformation)
            feats = feat_dict["dirty"]
            targets_other = targets_dict["clean"]

            # Run features through other decoder
            translated_feats = model.forward_decoder(feats, "clean")
            r_loss = reconstruction_loss(translated_feats, targets_other)
            
            decoder_class_losses.add("dirty", {"transformation_recon_loss": r_loss.data[0]})
        
        batches_processed += 1

    return decoder_class_losses



# Save model with best dev set loss thus far
best_dev_loss = float('inf')

# Regularize via patience-based early stopping
max_patience = 3
iterations_since_improvement = 0

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
        iterations_since_improvement = 0
        print("\nNew best dev set loss: %.6f" % best_dev_loss, flush=True)
        
        # Save a checkpoint for our model!
        state_obj = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_dev_loss": best_dev_loss,
            "dev_loss": dev_loss,
            "generator_optimizer": generator_optimizer.state_dict(),
        }
        torch.save(state_obj, ckpt_path)
        print("Saved checkpoint for model", flush=True)
    else:
        iterations_since_improvement += 1
        print("\nNo improvement in %d iterations (best dev set loss: %.6f)" % (iterations_since_improvement, best_dev_loss),
              flush=True)
        if iterations_since_improvement >= max_patience:
            print("STOPPING EARLY", flush=True)
            break

# Once done, load best checkpoint and determine reconstruction loss alone
print("Computing reconstruction loss...", flush=True)

# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
checkpoint = torch.load(ckpt_path, map_location=lambda storage,loc: storage)

# Set up model state and set to eval mode (i.e. disable batch norm)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Loaded checkpoint; best model ready now.")

train_loss_dict = test(epoch, training_loader)
print("\nTRAINING SET", flush=True)
print(train_loss_dict, flush=True)

dev_loss_dict = test(epoch, dev_loader)
print("\nDEV SET", flush=True)
print(dev_loss_dict, flush=True)

run_end_t = time.clock()
print("\nCompleted training run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
