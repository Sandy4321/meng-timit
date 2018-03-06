import math
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
from utils.kaldi_data import KaldiDataset

# Moved to function so that cProfile has a function to call
def run_training(run_mode, domain_adversarial, gan):
    run_start_t = time.clock()

    # Set up noising
    noise_ratio = float(os.environ["NOISE_RATIO"])
    print("Noising %.3f%% of input features" % (noise_ratio * 100.0), flush=True)

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
    
    use_backtranslation = True if os.environ["USE_BACKTRANSLATION"] == "true" else False
   
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

    # Set up training parameters
    batch_size = int(os.environ["BATCH_SIZE"])
    epochs = int(os.environ["EPOCHS"])
    optimizer_name = os.environ["OPTIMIZER"]
    learning_rate = float(os.environ["LEARNING_RATE"])
    l2_reg = float(os.environ["L2_REG"])

    on_gpu = torch.cuda.is_available()

    # Set up data files
    training_scps = dict()
    for decoder_class in decoder_classes:
        training_scp_name = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "train.scp")
        training_scps[decoder_class] = training_scp_name

    dev_scps = dict()
    for decoder_class in decoder_classes:
        dev_scp_name = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "dev.scp")
        dev_scps[decoder_class] = dev_scp_name

    # Fix random seed for debugging
    torch.manual_seed(1)
    if on_gpu:
        torch.cuda.manual_seed_all(1)
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
                                    decoder_classes=decoder_classes,
                                    weight_init=weight_init)
    elif run_mode == "vae":
        if domain_adversarial:
            print("Domain adversarial VAEs not supported yet", flush=True)
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
                                    decoder_classes=decoder_classes,
                                    weight_init=weight_init)
    else:
        print("Unknown train mode %s" % run_mode, flush=True)
        sys.exit(1)

    if on_gpu:
        model.cuda()

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
    print("Done constructing model.", flush=True)
    print(model, flush=True)

    # Count number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model has %d trainable parameters" % params, flush=True)

    # Set up loss functions
    # l2_criterion = nn.MSELoss(size_average=False)
    l2_criterion = nn.MSELoss()
    if on_gpu:
        l2_criterion.cuda()
    def reconstruction_loss(recon_x, x):
        L2 = l2_criterion(recon_x.view(-1, time_dim, freq_dim),
                          x.view(-1, time_dim, freq_dim))
        return L2

    def kld_loss(recon_x, x, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.size()[0] * feat_dim

        return KLD

    # Use BCEWithLogitsLoss to get better numerical stability
    # bce_criterion = nn.BCEWithLogitsLoss(size_average=False)
    bce_criterion = nn.BCEWithLogitsLoss()
    if on_gpu:
        bce_criterion.cuda()
    def discriminative_loss(guess_output, truth_class):
        return bce_criterion(guess_output, truth_class)



    # TRAIN MULTIDECODER



    # Set up optimizers for the generator portion (multidecoder itself)
    generator_optimizer = getattr(optim, optimizer_name)(model.generator_parameters(),
                                                         lr=learning_rate,
                                                         weight_decay=l2_reg)
    if domain_adversarial:
        # Set up optimizer for domain classifier adversary
        domain_adversary_optimizer = getattr(optim, optimizer_name)(model.domain_adversary_parameters(),
                                                                    lr=learning_rate,
                                                                    weight_decay=l2_reg)
    if gan:
        # Set up real/fake discriminators for each decoder
        gan_optimizers = dict()
        for decoder_class in decoder_classes:
            gan_optimizers[decoder_class] = getattr(optim, optimizer_name)(model.gan_parameters(decoder_class),
                                                                           lr=learning_rate,
                                                                           weight_decay=l2_reg)

    print("Setting up data...", flush=True)
    # loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}
    loader_kwargs = {"num_workers": 1}

    print("Setting up training datasets...", flush=True)
    training_datasets = dict()
    training_loaders = dict()
    for decoder_class in decoder_classes:
        current_dataset = KaldiDataset(training_scps[decoder_class],
                                     left_context=left_context,
                                     right_context=right_context,
                                     shuffle_utts=True,
                                     shuffle_feats=True)
        training_datasets[decoder_class] = current_dataset
        training_loaders[decoder_class] = DataLoader(current_dataset,
                                                     batch_size=batch_size,
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
        current_dataset = KaldiDataset(dev_scps[decoder_class],
                                     left_context=left_context,
                                     right_context=right_context,
                                     shuffle_utts=True,
                                     shuffle_feats=True)
        dev_datasets[decoder_class] = current_dataset
        dev_loaders[decoder_class] = DataLoader(current_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                **loader_kwargs)
        print("Using %d dev features (%d batches) for class %s" % (len(current_dataset),
                                                                   len(dev_loaders[decoder_class]),
                                                                   decoder_class),
              flush=True)

    # Set up minibatch shuffling (for training only) by decoder class
    print("Setting up minibatch shuffling for training...", flush=True)
    train_batch_counts = dict()
    train_element_counts = dict()
    dev_batch_counts = dict()
    dev_element_counts = dict()

    total_train_batches = float('inf')

    for decoder_class in decoder_classes:
        train_batch_counts[decoder_class] = len(training_loaders[decoder_class])
        train_element_counts[decoder_class] = len(training_datasets[decoder_class])
        dev_batch_counts[decoder_class] = len(dev_loaders[decoder_class])
        dev_element_counts[decoder_class] = len(dev_datasets[decoder_class])

        # If datasets mismatch in size, use the smaller of the two
        total_train_batches = min(train_batch_counts[decoder_class], total_train_batches)

    log_interval = total_train_batches // 10
        
    print("%d total batches: %s" % (total_train_batches, str(train_batch_counts)), flush=True)

    print("Done setting up data.", flush=True)

    # Utilities for loss dictionaries
    def print_loss_dict(loss_dict, class_elements_processed):
        print("Losses:", flush=True)
        for decoder_class in loss_dict:
            print("=> Class %s" % decoder_class, flush=True)
            class_loss = 0.0
            for loss_key in loss_dict[decoder_class]:
                current_loss = loss_dict[decoder_class][loss_key] / class_elements_processed[decoder_class]
                class_loss += current_loss
                print("===> %s: %.3f" % (loss_key, current_loss), flush=True)
            print("===> Total for class %s: %.3f" % (decoder_class, class_loss), flush=True)
        print("TOTAL: %.3f" % total_loss(loss_dict, class_elements_processed), flush=True)

    def total_loss(loss_dict, class_elements_processed):
        loss = 0.0
        for decoder_class in loss_dict:
            for loss_key in loss_dict[decoder_class]:
                current_loss = loss_dict[decoder_class][loss_key] / class_elements_processed[decoder_class]
                loss += current_loss
        return loss



    def train(epoch):
        model.train()

        training_iterators = {decoder_class: iter(training_loaders[decoder_class]) for decoder_class in decoder_classes}

        decoder_class_losses = {}
        for decoder_class in decoder_classes:
            decoder_class_losses[decoder_class] = {}

            decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
            if use_backtranslation:
                decoder_class_losses[decoder_class]["backtranslation_recon_loss"] = 0.0
            if run_mode == "vae":
                # Track KL divergence as well
                decoder_class_losses[decoder_class]["autoencoding_kld"] = 0.0
                if use_backtranslation:
                    decoder_class_losses[decoder_class]["backtranslation_kld"] = 0.0
            if domain_adversarial:
                decoder_class_losses[decoder_class]["domain_adversarial_loss"] = 0.0
            elif gan:
                decoder_class_losses[decoder_class]["real_gan_loss"] = 0.0
                decoder_class_losses[decoder_class]["fake_gan_loss"] = 0.0

        batches_processed = 0
        class_elements_processed = {decoder_class: 0 for decoder_class in decoder_classes}

        while batches_processed < total_train_batches:
            # Get data for each decoder class
            feat_dict = dict()
            targets_dict = dict()
            element_counts = dict()     # Used to get loss per element, rather than batch, in printed output
            for decoder_class in decoder_classes:
                feats, targets = training_iterators[decoder_class].next()
                element_counts[decoder_class] = feats.size()[0]

                if on_gpu:
                    feats = feats.cuda()
                    targets = targets.cuda()
                feats = Variable(feats)
                targets = Variable(targets)

                feat_dict[decoder_class] = feats
                targets_dict[decoder_class] = targets
            
            # Handle noising, if desired
            noised_feat_dict = dict()
            for decoder_class in decoder_classes:
                feats = feat_dict[decoder_class]
                if noise_ratio > 0.0:
                    # Add noise to signal; randomly drop out % of elements
                    noise_matrix = torch.FloatTensor(np.random.binomial(1, 1.0 - noise_ratio, size=feats.size()).astype(float))
                    
                    if on_gpu:
                        noise_matrix = noise_matrix.cuda()
                    noise_matrix = Variable(noise_matrix, requires_grad=False)
                    noised_feats = torch.mul(feats, noise_matrix)
                else:
                    noised_feats = feats
                noised_feat_dict[decoder_class] = noised_feats


            # STEP 1: Autoencoder training


            generator_optimizer.zero_grad()
            for i in range(len(decoder_classes)):
                # This is dumb with two classes, I know
                decoder_class = decoder_classes[i]
                other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]
                

                # PHASE 1: Backprop through same decoder (denoised reconstruction)


                noised_feats = noised_feat_dict[decoder_class]
                targets = targets_dict[decoder_class]

                if run_mode == "ae":
                    recon_batch = model.forward_decoder(noised_feats, decoder_class)
                elif run_mode == "vae":
                    recon_batch, mu, logvar = model.forward_decoder(noised_feats, decoder_class)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                if run_mode == "ae":
                    r_loss = reconstruction_loss(recon_batch, targets)
                    r_loss.backward(retain_graph=True)
                elif run_mode == "vae":
                    r_loss = reconstruction_loss(recon_batch, targets)
                    k_loss = kld_loss(recon_batch, targets, mu, logvar)
                    vae_loss = r_loss + k_loss
                    vae_loss.backward(retain_graph=True)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]
                if run_mode == "vae":
                    decoder_class_losses[decoder_class]["autoencoding_kld"] += k_loss.data[0]
            
                if use_backtranslation:
                    # PHASE 2: Backtranslation


                    # Run (unnoised) features through other decoder
                    feats = feat_dict[decoder_class]

                    if run_mode == "ae":
                        translated_feats = model.forward_decoder(feats, other_decoder_class)
                    elif run_mode == "vae":
                        translated_feats, translated_mu, translated_logvar = model.forward_decoder(feats, other_decoder_class)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    # Run translated features back through original decoder
                    if run_mode == "ae":
                        recon_translated_batch = model.forward_decoder(translated_feats, decoder_class)
                    elif run_mode == "vae":
                        recon_translated_batch, mu, logvar = model.forward_decoder(translated_feats, decoder_class)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)

                    if run_mode == "ae":
                        r_loss = reconstruction_loss(recon_translated_batch, targets)
                        r_loss.backward(retain_graph=True)
                    elif run_mode == "vae":
                        r_loss = reconstruction_loss(recon_translated_batch, targets)
                        k_loss = kld_loss(recon_translated_batch, targets, mu, logvar)
                        vae_loss = r_loss + k_loss
                        vae_loss.backward(retain_graph=True)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    decoder_class_losses[decoder_class]["backtranslation_recon_loss"] += r_loss.data[0]
                    if run_mode == "vae":
                        decoder_class_losses[decoder_class]["backtranslation_kld"] += k_loss.data[0]

            # Now that we've seen both classes, update generator
            generator_optimizer.step()


            # STEP 2: Adversarial training


            if domain_adversarial:
                # Train just domain_adversary
                domain_adversary_optimizer.zero_grad()
                disc_losses = dict()
                for decoder_class in decoder_classes:
                    feats = feat_dict[decoder_class]

                    if run_mode == "ae":
                        latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                                       1,
                                                                                                       time_dim,
                                                                                                       freq_dim))
                    elif run_mode == "vae":
                        print("Domain adversarial VAEs not supported yet", flush=True)
                        sys.exit(1)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    class_prediction = model.domain_adversary.forward(latent)
                    class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "clean" else torch.FloatTensor(np.ones(class_prediction.size()))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth)

                    disc_loss = discriminative_loss(class_prediction, class_truth)
                    disc_loss.backward(retain_graph=True)
                    disc_losses[decoder_class] = disc_loss
                
                # Now that we've seen both classes, update the adversary
                domain_adversary_optimizer.step()
                        
                # Train just encoder (no loss prop'ed through decoders anyway),
                # using negative discriminative loss instead
                generator_optimizer.zero_grad()
                for decoder_class in decoder_classes:
                    disc_loss = disc_losses[decoder_class]
                    domain_adv_loss = -disc_loss
                    domain_adv_loss.backward(retain_graph=True)
                    decoder_class_losses[decoder_class]["domain_adversarial_loss"] += domain_adv_loss.data[0]

                # Now that we've seen both classes, update the encoder
                generator_optimizer.step()
            elif gan:
                # Generative adversarial loss
                # Adversary determines whether output is real data, or TRANSFORMED data
                # i.e., is this example real SDM1 data, or IHM data decoded into SDM1 via multidecoder?
                # Convention for classifier: 1 is True, 0 is Fake

                for decoder_class in decoder_classes:
                    gan_optimizers[decoder_class].zero_grad()

                # Train just discriminators
                real_disc_losses = dict()   # Store these so we don't need to compute them again
                fake_disc_losses = dict()   # Store these so we don't need to compute them again
                for i in range(len(decoder_classes)):
                    # This is dumb with two classes, I know
                    decoder_class = decoder_classes[i]
                    other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]

                    feats = feat_dict[decoder_class] 

                    if run_mode == "ae":
                        # Create minibatch of transformed examples
                        sim_feats = model.forward_decoder(feats, other_decoder_class)
                    elif run_mode == "vae":
                        print("Generative adversarial VAEs not supported yet", flush=True)
                        sys.exit(1)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    # Real examples
                    class_prediction = model.forward_gan(feats, decoder_class)
                    class_truth = torch.FloatTensor(np.ones((class_prediction.size()[0], 1)))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth)

                    real_disc_loss = discriminative_loss(class_prediction, class_truth)
                    real_disc_loss.backward(retain_graph=True)
                    real_disc_losses[decoder_class] = real_disc_loss
                    
                    # Fake examples
                    class_prediction = model.forward_gan(sim_feats, other_decoder_class)
                    class_truth = torch.FloatTensor(np.zeros((class_prediction.size()[0], 1)))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth)

                    fake_disc_loss = discriminative_loss(class_prediction, class_truth)
                    fake_disc_loss.backward(retain_graph=True)
                    fake_disc_losses[other_decoder_class] = real_disc_loss

                for decoder_class in decoder_classes:
                    # Update discriminators, now that we've seen real and fake examples for both
                    gan_optimizers[decoder_class].step()
                
                # Train encoder + decoders, w/ negative discriminative losses
                generator_optimizer.zero_grad()
                for i in range(len(decoder_classes)):
                    # This is dumb with two classes, I know
                    decoder_class = decoder_classes[i]
                    other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]

                    # Real examples
                    real_disc_loss = real_disc_losses[decoder_class]
                    real_adv_loss = -real_disc_loss
                    real_adv_loss.backward(retain_graph=True)

                    decoder_class_losses[decoder_class]["real_gan_loss"] += real_adv_loss.data[0]                
                    
                    # Fake examples
                    fake_disc_loss = fake_disc_losses[other_decoder_class]
                    fake_adv_loss = -fake_disc_loss
                    fake_adv_loss.backward(retain_graph=True)

                    decoder_class_losses[other_decoder_class]["fake_gan_loss"] += fake_adv_loss.data[0]                
                # Update decoders and encoder, now that we've seen real and fake examples for both
                generator_optimizer.step()

            # Print updates, if any
            batches_processed += 1
            for decoder_class in decoder_classes:
                class_elements_processed[decoder_class] += element_counts[decoder_class]

            if batches_processed % log_interval == 0:
                print("Train epoch %d: [%d/%d (%.1f%%)]" % (epoch,
                                                            batches_processed,
                                                            total_train_batches,
                                                            batches_processed / total_train_batches * 100.0),
                      flush=True)
                print_loss_dict(decoder_class_losses, class_elements_processed) 
        return decoder_class_losses

    def test(epoch, loaders, recon_only=False, noised=True):
        decoder_class_losses = {}
        for decoder_class in decoder_classes:
            decoder_class_losses[decoder_class] = {}

            decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
            if use_backtranslation:
                decoder_class_losses[decoder_class]["backtranslation_recon_loss"] = 0.0
            if run_mode == "vae" and not recon_only:
                # Track KL divergence as well
                decoder_class_losses[decoder_class]["autoencoding_kld"] = 0.0
                if use_backtranslation:
                    decoder_class_losses[decoder_class]["backtranslation_kld"] = 0.0
            if domain_adversarial and not recon_only:
                decoder_class_losses[decoder_class]["domain_adversarial_loss"] = 0.0
            elif gan and not recon_only:
                decoder_class_losses[decoder_class]["real_gan_loss"] = 0.0
                decoder_class_losses[decoder_class]["fake_gan_loss"] = 0.0

        other_decoder_class = decoder_classes[1]
        for decoder_class in decoder_classes:
            for feats, targets in loaders[decoder_class]:
                # Set to volatile so history isn't saved (i.e., not training time)
                if on_gpu:
                    feats = feats.cuda()
                    targets = targets.cuda()
                feats = Variable(feats, volatile=True)
                targets = Variable(targets, volatile=True)
            
                # Set up noising, if needed
                if noised:
                    if noise_ratio > 0.0:
                        # Add noise to signal; randomly drop out % of elements
                        noise_matrix = torch.FloatTensor(np.random.binomial(1, 1.0 - noise_ratio, size=feats.size()).astype(float))
                        if on_gpu:
                            noise_matrix = noise_matrix.cuda()
                        noise_matrix = Variable(noise_matrix, requires_grad=False, volatile=True)
                        noised_feats = torch.mul(feats, noise_matrix)
                    else:
                        noised_feats = feats.clone()

                # PHASE 1: Backprop through same decoder (denoised autoencoding)
                model.eval()
                if run_mode == "ae":
                    if noised:
                        recon_batch = model.forward_decoder(noised_feats, decoder_class)
                    else:
                        recon_batch = model.forward_decoder(feats, decoder_class)
                elif run_mode == "vae":
                    if noised:
                        recon_batch, mu, logvar = model.forward_decoder(noised_feats, decoder_class)
                    else:
                        recon_batch, mu, logvar = model.forward_decoder(feats, decoder_class)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                if run_mode == "ae" or recon_only:
                    r_loss = reconstruction_loss(recon_batch, targets)
                elif run_mode == "vae":
                    r_loss = reconstruction_loss(recon_batch, targets)
                    k_loss = kld_loss(recon_batch, targets, mu, logvar)
                    vae_loss = r_loss + k_loss
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]
                if run_mode == "vae" and not recon_only:
                    decoder_class_losses[decoder_class]["autoencoding_kld"] += k_loss.data[0]

                if use_backtranslation:
                    # PHASE 2: Backtranslation

                    # Run (unnoised) features through other decoder
                    if run_mode == "ae":
                        translated_feats = model.forward_decoder(feats, other_decoder_class)
                    elif run_mode == "vae":
                        translated_feats, translated_mu, translated_logvar = model.forward_decoder(feats, other_decoder_class)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    # Run translated features back through original decoder
                    if run_mode == "ae":
                        recon_translated_batch = model.forward_decoder(translated_feats, decoder_class)
                    elif run_mode == "vae":
                        recon_translated_batch, mu, logvar = model.forward_decoder(translated_feats, decoder_class)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)

                    if run_mode == "ae" or recon_only:
                        r_loss = reconstruction_loss(recon_translated_batch, targets)
                    elif run_mode == "vae":
                        r_loss = reconstruction_loss(recon_translated_batch, targets)
                        k_loss = kld_loss(recon_translate_batch, targets, mu, logvar)
                        vae_loss = r_loss + k_loss
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    decoder_class_losses[decoder_class]["backtranslation_recon_loss"] += r_loss.data[0]
                    if run_mode == "vae" and not recon_only:
                        decoder_class_losses[decoder_class]["backtranslation_kld"] += k_loss.data[0]
               
                if domain_adversarial and not recon_only:
                    # Domain adversarial loss
                    if run_mode == "ae":
                        latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                                       1,
                                                                                                       time_dim,
                                                                                                       freq_dim))
                    elif run_mode == "vae":
                        print("Domain adversarial VAEs not supported yet", flush=True)
                        sys.exit(1)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    class_prediction = model.domain_adversary.forward(latent)
                    class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "clean" else torch.FloatTensor(np.ones(class_prediction.size()))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)
                    disc_loss = discriminative_loss(class_prediction, class_truth)
                    domain_adv_loss = -disc_loss
                    
                    decoder_class_losses[decoder_class]["domain_adversarial_loss"] += domain_adv_loss.data[0]
                elif gan and not recon_only:
                    # Generative adversarial loss
                    # Adversary determines whether output is real data, or TRANSFORMED data
                    # i.e., is this example real SDM1 data, or IHM data decoded into SDM1 via multidecoder?
                    if run_mode == "ae":
                        # Create minibatch of transformed examples
                        sim_feats = model.forward_decoder(feats, other_decoder_class)
                    elif run_mode == "vae":
                        print("Generative adversarial VAEs not supported yet", flush=True)
                        sys.exit(1)
                    else:
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                
                    # Real examples
                    class_prediction = model.forward_gan(feats, decoder_class)
                    class_truth = torch.FloatTensor(np.ones((class_prediction.size()[0], 1)))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)

                    real_disc_loss = discriminative_loss(class_prediction, class_truth)
                    real_adv_loss = -real_disc_loss
                    decoder_class_losses[decoder_class]["real_gan_loss"] += real_adv_loss.data[0]                
                    
                    # Fake examples
                    class_prediction = model.forward_gan(sim_feats, other_decoder_class)
                    class_truth = torch.FloatTensor(np.zeros((class_prediction.size()[0], 1)))
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)

                    fake_disc_loss = discriminative_loss(class_prediction, class_truth)
                    fake_adv_loss = -fake_disc_loss
                    decoder_class_losses[other_decoder_class]["fake_gan_loss"] += fake_adv_loss.data[0]                
            other_decoder_class = decoder_class
            
        return decoder_class_losses

    # Save model with best dev set loss thus far
    best_dev_loss = float('inf')
    save_best_only = True   # Set to False to always save model state, regardless of improvement

    def save_checkpoint(state_obj, is_best, model_dir):
        if not save_best_only:
            if domain_adversarial:
                ckpt_path = os.path.join(model_dir, "ckpt_cnn_domain_adversarial_fc_%s_act_%s_%s_ratio%s_md_%d.pth.tar" % (os.environ["DOMAIN_ADV_FC_DELIM"],
                                                                                                        domain_adv_activation,
                                                                                                        run_mode,
                                                                                                        str(noise_ratio),
                                                                                                        state_obj["epoch"]))
            elif gan:
                ckpt_path = os.path.join(model_dir, "ckpt_cnn_gan_fc_%s_act_%s_%s_ratio%s_md_%d.pth.tar" % (os.environ["GAN_FC_DELIM"],
                                                                                                        gan_activation,
                                                                                                        run_mode,
                                                                                                        str(noise_ratio),
                                                                                                        state_obj["epoch"]))
            else:
                ckpt_path = os.path.join(model_dir, "ckpt_cnn_%s_ratio%s_md_%d.pth.tar" % (run_mode, str(noise_ratio), state_obj["epoch"]))

            torch.save(state_obj, ckpt_path)
            if is_best:
                shutil.copyfile(ckpt_path, best_ckpt_path)
        else:
            torch.save(state_obj, best_ckpt_path)

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
        print_loss_dict(train_loss_dict, train_element_counts)
                
        dev_loss_dict = test(epoch, dev_loaders)
        print("\nEPOCH %d DEV" % epoch, flush=True)
        print_loss_dict(dev_loss_dict, dev_element_counts)

        dev_loss = total_loss(dev_loss_dict, dev_element_counts)
        is_best = (dev_loss <= best_dev_loss)
        if is_best:
            best_dev_loss = dev_loss
            iterations_since_improvement = 0
            print("\nNew best dev set loss: %.6f" % best_dev_loss, flush=True)
        else:
            iterations_since_improvement += 1
            print("\nNo improvement in %d iterations (best dev set loss: %.6f)" % (iterations_since_improvement, best_dev_loss),
                  flush=True)
            if iterations_since_improvement >= max_patience:
                print("STOPPING EARLY", flush=True)
                break

        if not save_best_only or (save_best_only and is_best):
            # Save a checkpoint for our model!
            state_obj = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_dev_loss": best_dev_loss,
                "dev_loss": dev_loss,
                "generator_optimizer": generator_optimizer.state_dict(),
            }
            if domain_adversarial:
                state_obj["domain_adversary_optimizer"] = domain_adversary_optimizer.state_dict()
            elif gan:
                state_obj["gan_optimizers"] = {decoder_class: gan_optimizers[decoder_class].state_dict() for decoder_class in decoder_classes}

            save_checkpoint(state_obj, is_best, model_dir)
            print("Saved checkpoint for model", flush=True)
        else:
            print("Not saving checkpoint; no improvement made", flush=True)
        
        train_end_t = time.clock()
        print("\nEPOCH %d (%.3fs)" % (epoch,
                                      train_end_t - train_start_t),
              flush=True)

    # Once done, load best checkpoint and determine reconstruction loss alone
    print("Computing reconstruction loss...", flush=True)

    # Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
    checkpoint = torch.load(best_ckpt_path, map_location=lambda storage,loc: storage)

    # Set up model state and set to eval mode (i.e. disable batch norm)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Loaded checkpoint; best model ready now.")

    train_loss_dict = test(epoch, training_loaders, recon_only=True, noised=False)
    print("\nTRAINING SET", flush=True)
    print_loss_dict(train_loss_dict, train_element_counts)

    dev_loss_dict = test(epoch, dev_loaders, recon_only=True, noised=False)
    print("\nDEV SET", flush=True)
    print_loss_dict(dev_loss_dict, dev_element_counts)

    run_end_t = time.clock()
    print("\nCompleted training run in %.3f seconds" % (run_end_t - run_start_t), flush=True)



# Parse command line args
run_mode = "ae"
domain_adversarial = False
gan = False
profile = False

if len(sys.argv) >= 4:
    run_mode = sys.argv[1]
    domain_adversarial = True if sys.argv[2] == "true" else False
    gan = True if sys.argv[3] == "true" else False
    if len(sys.argv) >= 5:
        profile = True if sys.argv[4] == "profile" else False
else:
    print("Usage: python cnn/scripts/train_md.py <run mode> <domain_adversarial true/false> <GAN true/false> <profile (optional)>", flush=True)
    sys.exit(1)

print("Running training with mode %s" % run_mode, flush=True)
if domain_adversarial:
    print("Using domain_adversarial loss", flush=True)
elif gan:
    print("Using generative adversarial loss", flush=True)

if profile:
    print("Profiling code using cProfile", flush=True)

    import cProfile
    profile_output_dir = os.path.join(os.environ["LOG_DIR"], os.environ["EXPT_NAME"])
    profile_output_path = os.path.join(profile_output_dir, "train_%s.prof" % run_mode)
    cProfile.run('run_training(run_mode, domain_adversarial, gan)', profile_output_path) 
else:
    run_training(run_mode, domain_adversarial, gan)

