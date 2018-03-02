import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./cnn")
from cnn_md import CNNMultidecoder, CNNVariationalMultidecoder
from cnn_md import CNNDomainAdversarialMultidecoder
from cnn_md import CNNGANMultidecoder
from utils.kaldi_data import KaldiEvalDataset, write_kaldi_ark, write_kaldi_scp

run_start_t = time.clock()

# Parse command line args
run_mode = "ae"
domain_adversarial = False
gan = False

if len(sys.argv) == 4:
    run_mode = sys.argv[1]
    domain_adversarial = True if sys.argv[2] == "true" else False
    gan = True if sys.argv[3] == "true" else False
else:
    print("Usage: python cnn/scripts/augment_md.py <run mode> <domain_adversarial true/false> <GAN true/false>", flush=True)
    sys.exit(1)

print("Running augmentation with mode %s" % run_mode, flush=True)
if domain_adversarial:
    print("Using domain_adversarial loss", flush=True)
elif gan:
    print("Using generative adversarial loss", flush=True)

# Set up noising
noise_ratio = float(os.environ["NOISE_RATIO"])
print("Noise ratio: %.3f%% of input features" % (noise_ratio * 100.0), flush=True)

# Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py

# Set up features
feature_name = "fbank"
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])

freq_dim = feat_dim
time_dim = (left_context + right_context + 1)

# Read in parameters to use for our network
enc_channel_sizes = []
for res_str in os.environ["ENC_CHANNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_channel_sizes.append(int(res_str))
enc_kernel_sizes = []
for res_str in os.environ["ENC_KERNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_kernel_sizes.append(int(res_str))
enc_downsample_sizes = []
for res_str in os.environ["ENC_DOWNSAMPLES_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_downsample_sizes.append(int(res_str))
enc_fc_sizes = []
for res_str in os.environ["ENC_FC_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_fc_sizes.append(int(res_str))

latent_dim = int(os.environ["LATENT_DIM"])

dec_fc_sizes = []
for res_str in os.environ["DEC_FC_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_fc_sizes.append(int(res_str))
dec_channel_sizes = []
for res_str in os.environ["DEC_CHANNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_channel_sizes.append(int(res_str))
dec_kernel_sizes = []
for res_str in os.environ["DEC_KERNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_kernel_sizes.append(int(res_str))
dec_upsample_sizes = []
for res_str in os.environ["DEC_UPSAMPLES_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_upsample_sizes.append(int(res_str))

activation = os.environ["ACTIVATION_FUNC"]
decoder_classes = []
for res_str in os.environ["DECODER_CLASSES_DELIM"].split("_"):
    if len(res_str) > 0:
        decoder_classes.append(res_str)
use_batch_norm = True if os.environ["USE_BATCH_NORM"] == "true" else False
weight_init = os.environ["WEIGHT_INIT"]

strided = True if os.environ["STRIDED"] == "true" else False
    
if domain_adversarial:
    domain_adv_fc_sizes = []
    for res_str in os.environ["DOMAIN_ADV_FC_DELIM"].split("_"):
        if len(res_str) > 0:
            domain_adv_fc_sizes.append(int(res_str))
    domain_adv_activation = os.environ["DOMAIN_ADV_ACTIVATION"]
    
if gan:
    gan_fc_sizes = []
    for res_str in os.environ["GAN_FC_DELIM"].split("_"):
        if len(res_str) > 0:
            gan_fc_sizes.append(int(res_str))
    gan_activation = os.environ["GAN_ACTIVATION"]

on_gpu = torch.cuda.is_available()
log_interval = 100   # Log results once for this many batches during training

# Set up input files and output directory
training_scps = dict()
for decoder_class in decoder_classes:
    training_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-train-norm.blogmel.scp" % decoder_class)
    training_scps[decoder_class] = training_scp_name

dev_scps = dict()
for decoder_class in decoder_classes:
    dev_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-dev-norm.blogmel.scp" % decoder_class)
    dev_scps[decoder_class] = dev_scp_name

if domain_adversarial:
    output_dir = os.path.join(os.environ["AUGMENTED_DATA_DIR"], "domain_adversarial_fc_%s_act_%s_%s_ratio%s" % (os.environ["DOMAIN_ADV_FC_DELIM"],
                                                                                             domain_adv_activation,
                                                                                             run_mode,
                                                                                             str(noise_ratio)))
elif gan:
    output_dir = os.path.join(os.environ["AUGMENTED_DATA_DIR"], "gan_fc_%s_act_%s_%s_ratio%s" % (os.environ["GAN_FC_DELIM"],
                                                                                             gan_activation,
                                                                                             run_mode,
                                                                                             str(noise_ratio)))
else:
    output_dir = os.path.join(os.environ["AUGMENTED_DATA_DIR"], "%s_ratio%s" % (run_mode, noise_ratio))

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)
random.seed(1)

# Construct autoencoder with our parameters
print("Constructing model...", flush=True)
if run_mode == "ae":
    if domain_adversarial:
        model = CNNDomainAdversarialMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                enc_channel_sizes=enc_channel_sizes,
                                enc_kernel_sizes=enc_kernel_sizes,
                                enc_downsample_sizes=enc_downsample_sizes,
                                enc_fc_sizes=enc_fc_sizes,
                                latent_dim=latent_dim,
                                dec_fc_sizes=dec_fc_sizes,
                                dec_channel_sizes=dec_channel_sizes,
                                dec_kernel_sizes=dec_kernel_sizes,
                                dec_upsample_sizes=dec_upsample_sizes,
                                activation=activation,
                                use_batch_norm=use_batch_norm,
                                strided=strided,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init,
                                domain_adv_fc_sizes=domain_adv_fc_sizes,
                                domain_adv_activation=domain_adv_activation)
    elif gan:
        model = CNNGANMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                enc_channel_sizes=enc_channel_sizes,
                                enc_kernel_sizes=enc_kernel_sizes,
                                enc_downsample_sizes=enc_downsample_sizes,
                                enc_fc_sizes=enc_fc_sizes,
                                latent_dim=latent_dim,
                                dec_fc_sizes=dec_fc_sizes,
                                dec_channel_sizes=dec_channel_sizes,
                                dec_kernel_sizes=dec_kernel_sizes,
                                dec_upsample_sizes=dec_upsample_sizes,
                                activation=activation,
                                use_batch_norm=use_batch_norm,
                                strided=strided,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init,
                                gan_fc_sizes=gan_fc_sizes,
                                gan_activation=gan_activation)
    else:
        model = CNNMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                enc_channel_sizes=enc_channel_sizes,
                                enc_kernel_sizes=enc_kernel_sizes,
                                enc_downsample_sizes=enc_downsample_sizes,
                                enc_fc_sizes=enc_fc_sizes,
                                latent_dim=latent_dim,
                                dec_fc_sizes=dec_fc_sizes,
                                dec_channel_sizes=dec_channel_sizes,
                                dec_kernel_sizes=dec_kernel_sizes,
                                dec_upsample_sizes=dec_upsample_sizes,
                                activation=activation,
                                use_batch_norm=use_batch_norm,
                                strided=strided,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
elif run_mode == "vae":
    if domain_adversarial:
        print("Adversarial VAEs not supported yet", flush=True)
        sys.exit(1)
    elif gan:
        print("Generative adversarial VAEs not supported yet", flush=True)
        sys.exit(1)
    else:
        model = CNNVariationalMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                enc_channel_sizes=enc_channel_sizes,
                                enc_kernel_sizes=enc_kernel_sizes,
                                enc_downsample_sizes=enc_downsample_sizes,
                                enc_fc_sizes=enc_fc_sizes,
                                latent_dim=latent_dim,
                                dec_fc_sizes=dec_fc_sizes,
                                dec_channel_sizes=dec_channel_sizes,
                                dec_kernel_sizes=dec_kernel_sizes,
                                dec_upsample_sizes=dec_upsample_sizes,
                                activation=activation,
                                use_batch_norm=use_batch_norm,
                                strided=strided,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
else:
    print("Unknown train mode %s" % run_mode, flush=True)
    sys.exit(1)

if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
print("Loading checkpoint...")
model_dir = os.environ["MODEL_DIR"]
if domain_adversarial:
    best_ckpt_path = os.path.join(model_dir, "best_cnn_domain_adversarial_fc_%s_act_%s_%s_ratio%s_md.pth.tar" % (os.environ["DOMAIN_ADV_FC_DELIM"],
                                                                                              domain_adv_activation,
                                                                                              run_mode,
                                                                                              str(noise_ratio)))
elif gan:
    best_ckpt_path = os.path.join(model_dir, "best_cnn_gan_fc_%s_act_%s_%s_ratio%s_md.pth.tar" % (os.environ["GAN_FC_DELIM"],
                                                                                              gan_activation,
                                                                                              run_mode,
                                                                                              str(noise_ratio)))
else:
    best_ckpt_path = os.path.join(model_dir, "best_cnn_%s_ratio%s_md.pth.tar" % (run_mode, str(noise_ratio)))
checkpoint = torch.load(best_ckpt_path, map_location=lambda storage,loc: storage)

# Set up model state and set to eval mode (i.e. disable batch norm)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Loaded checkpoint; best model ready now.")



# PERFORM DATA AUGMENTATION USING MULTIDECODER



print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up training datasets...", flush=True)
training_datasets = dict()
training_loaders = dict()
for decoder_class in decoder_classes:
    current_dataset = KaldiEvalDataset(training_scps[decoder_class])
    training_datasets[decoder_class] = current_dataset
    training_loaders[decoder_class] = DataLoader(current_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 **loader_kwargs)
    print("Using %d training features (%d batches) for class %s" % (len(current_dataset),
                                                                    len(training_loaders[decoder_class]),
                                                                    decoder_class),
          flush=True)

print("Setting up dev datasets...", flush=True)
dev_datasets = dict()
dev_loaders = dict()
for decoder_class in decoder_classes:
    current_dataset = KaldiEvalDataset(dev_scps[decoder_class])
    dev_datasets[decoder_class] = current_dataset
    dev_loaders[decoder_class] = DataLoader(current_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
    print("Using %d dev features (%d batches) for class %s" % (len(current_dataset),
                                                               len(dev_loaders[decoder_class]),
                                                               decoder_class),
          flush=True)

print("Done setting up data.", flush=True)

def augment(source_class, target_class):
    model.eval()

    # Process training dataset
    print("=> Processing training data...", flush=True)
    batches_processed = 0
    total_batches = len(training_loaders[source_class])
    with open(os.path.join(output_dir, "train-src_%s-tar_%s.ark" % (source_class, target_class)), 'w') as ark_fd:
        for batch_idx, (feats, targets, utt_ids) in enumerate(training_loaders[source_class]):
            utt_id = utt_ids[0]     # Batch size 1; only one utterance

            # Run batch through target decoder
            feats_numpy = feats.numpy().reshape((-1, freq_dim))
            num_frames = feats_numpy.shape[0]
            decoded_feats = np.empty((num_frames, freq_dim))
            for i in range(num_frames):
                frame_spliced = np.zeros((time_dim, freq_dim))
                frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
                frame_spliced[left_context, :] = feats_numpy[i, :]
                frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]
                frame_tensor = Variable(torch.FloatTensor(frame_spliced))
                if on_gpu:
                    frame_tensor = frame_tensor.cuda()

                if run_mode == "ae":
                    recon_frames = model.forward_decoder(frame_tensor, target_class)
                elif run_mode == "vae":
                    recon_frames, mu, logvar = model.forward_decoder(frame_tensor, target_class)
                else:
                    print("Unknown augment mode %s" % run_mode, flush=True)
                    sys.exit(1)

                recon_frames_numpy = recon_frames.cpu().data.numpy().reshape((-1, freq_dim))
                decoded_feats[i, :] = recon_frames_numpy[left_context:left_context + 1, :]

            # Write to output file
            aug_utt_id = "src_%s_tar_%s_%s" % (source_class, target_class, utt_id)
            write_kaldi_ark(ark_fd, aug_utt_id, decoded_feats)

            batches_processed += 1
            if batches_processed % log_interval == 0:
                print("===> Augmented %d/%d batches (%.1f%%)]" % (batches_processed,
                                                                  total_batches,
                                                                  100.0 * batches_processed / total_batches),
                      flush=True)

    # Create corresponding SCP file
    print("===> Writing SCP...", flush=True)
    with open(os.path.join(output_dir, "train-src_%s-tar_%s.scp" % (source_class, target_class)), 'w') as scp_fd:
        write_kaldi_scp(scp_fd, os.path.join(output_dir, "train-src_%s-tar_%s.ark" % (source_class, target_class)))
    print("=> Done with training data", flush=True)

    # Process dev dataset
    print("=> Processing dev data...", flush=True)
    batches_processed = 0
    total_batches = len(dev_loaders[source_class])
    with open(os.path.join(output_dir, "dev-src_%s-tar_%s.ark" % (source_class, target_class)), 'w') as ark_fd:
        for batch_idx, (feats, targets, utt_ids) in enumerate(dev_loaders[source_class]):
            utt_id = utt_ids[0]     # Batch size 1; only one utterance

            # Run batch through target decoder
            feats_numpy = feats.numpy().reshape((-1, freq_dim))
            num_frames = feats_numpy.shape[0]
            decoded_feats = np.empty((num_frames, freq_dim))
            for i in range(num_frames):
                frame_spliced = np.zeros((time_dim, freq_dim))
                frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
                frame_spliced[left_context, :] = feats_numpy[i, :]
                frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]
                frame_tensor = Variable(torch.FloatTensor(frame_spliced))
                if on_gpu:
                    frame_tensor = frame_tensor.cuda()

                if run_mode == "ae":
                    recon_frames = model.forward_decoder(frame_tensor, target_class)
                elif run_mode == "vae":
                    recon_frames, mu, logvar = model.forward_decoder(frame_tensor, target_class)
                else:
                    print("Unknown augment mode %s" % run_mode, flush=True)
                    sys.exit(1)

                recon_frames_numpy = recon_frames.cpu().data.numpy().reshape((-1, freq_dim))
                decoded_feats[i, :] = recon_frames_numpy[left_context:left_context + 1, :]

            # Write to output file
            aug_utt_id = "src_%s_tar_%s_%s" % (source_class, target_class, utt_id)
            write_kaldi_ark(ark_fd, aug_utt_id, decoded_feats)

            batches_processed += 1
            if batches_processed % log_interval == 0:
                print("===> Augmented %d/%d batches (%.1f%%)]" % (batches_processed,
                                                                  total_batches,
                                                                  100.0 * batches_processed / total_batches),
                      flush=True)

    # Create corresponding SCP file
    print("===> Writing SCP...", flush=True)
    with open(os.path.join(output_dir, "dev-src_%s-tar_%s.scp" % (source_class, target_class)), 'w') as scp_fd:
        write_kaldi_scp(scp_fd, os.path.join(output_dir, "dev-src_%s-tar_%s.ark" % (source_class, target_class)))
    print("=> Done with dev data", flush=True)

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

# Go through each combo of source and target class
for source_class in decoder_classes:
    for target_class in decoder_classes:
        process_start_t = time.clock()
        print("PROCESSING SOURCE %s, TARGET %s" % (source_class, target_class), flush=True)
        augment(source_class, target_class)
        process_end_t = time.clock()
        print("PROCESSED SOURCE %s, TARGET %s IN %.3f SECONDS" % (source_class, target_class, process_end_t - process_start_t), flush=True)

run_end_t = time.clock()
print("Completed data augmentation run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
