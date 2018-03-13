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

from loss_dict import *
from setup_md import setup_model
from ckpt_md import best_ckpt_path, save_ckpt
from utils.kaldi_data import KaldiDataset

# Moved to function so that cProfile has a function to call
def run_training(run_mode, domain_adversarial, gan):
    run_start_t = time.clock()

    # Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py

    # Set up features
    feature_name = "fbank"
    feat_dim = int(os.environ["FEAT_DIM"])
    left_context = int(os.environ["LEFT_CONTEXT"])
    right_context = int(os.environ["RIGHT_CONTEXT"])

    freq_dim = feat_dim
    time_dim = (left_context + right_context + 1)
 
    # Set up training parameters
    batch_size = int(os.environ["BATCH_SIZE"])
    epochs = int(os.environ["EPOCHS"])
    optimizer_name = os.environ["OPTIMIZER"]
    learning_rate = float(os.environ["LEARNING_RATE"])
    l2_reg = float(os.environ["L2_REG"])
    use_reconstruction = True if os.environ["USE_RECONSTRUCTION"] == "true" else False
    use_transformation = True if os.environ["USE_TRANSFORMATION"] == "true" else False
    on_gpu = torch.cuda.is_available()

    # Fix random seed for debugging
    torch.manual_seed(1)
    if on_gpu:
        torch.cuda.manual_seed_all(1)
    random.seed(1)

    # Set up the model
    model = setup_model(run_mode, domain_adversarial, gan)
    if on_gpu:
        model.cuda()
    ckpt_path = best_ckpt_path(run_mode, domain_adversarial, gan)

    # Set up data files
    training_scps = dict()
    for decoder_class in decoder_classes:
        training_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "train")
        # training_scp_name = os.path.join(training_scp_dir, "feats.scp")
        # training_scp_name = os.path.join(training_scp_dir, "feats-norm.scp")
        training_scp_name = os.path.join(training_scp_dir, "feats-norm-small.scp")
        training_scps[decoder_class] = training_scp_name

    dev_scps = dict()
    for decoder_class in decoder_classes:
        dev_scp_dir = os.path.join(os.environ["%s_FEATS" % decoder_class.upper()], "dev")
        # dev_scp_name = os.path.join(dev_scp_dir, "feats.scp")
        # dev_scp_name = os.path.join(dev_scp_dir, "feats-norm.scp")
        dev_scp_name = os.path.join(dev_scp_dir, "feats-norm-small.scp")
        dev_scps[decoder_class] = dev_scp_name

    # Set up loss functions
    l2_criterion = nn.MSELoss(size_average=False)
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
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

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
    train_element_counts = dict()
    dev_element_counts = dict()

    total_train_batches = float('inf')
    for decoder_class in decoder_classes:
        train_element_counts[decoder_class] = len(training_datasets[decoder_class])
        dev_element_counts[decoder_class] = len(dev_datasets[decoder_class])

        # If datasets mismatch in size, use the smaller of the two so we don't overrun the end
        total_train_batches = min(len(training_loaders[decoder_class]), total_train_batches)

    # Set up logging at a reasonable interval
    # log_interval = max(1, total_train_batches // 10)
    # TODO: change this back once debugging done
    log_interval = 1
    total_train_batches = 1
        
    print("%d total batches" % total_train_batches, flush=True)

    print("Done setting up data.", flush=True)



    def train(epoch):
        model.train()

        # Set up some bookkeeping
        training_iterators = {decoder_class: iter(training_loaders[decoder_class]) for decoder_class in decoder_classes}
        decoder_class_losses = loss_dict()
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
            

            # STEP 1: Generator training


            if use_reconstruction:
                # PHASE 1: Backprop through same decoder (reconstruction)
                generator_optimizer.zero_grad()

                # for i in range(len(decoder_classes)):
                # TODO: remove once done debugging!!!
                for i in range(1):
                    # This is dumb with two classes, I know
                    decoder_class = decoder_classes[i]
                    other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]

                    feats = feat_dict[decoder_class]
                    targets = targets_dict[decoder_class]

                    recon_batch = model.forward_decoder(feats, decoder_class)

                    r_loss = reconstruction_loss(recon_batch, targets)
                    r_loss.backward(retain_graph=True)

                    decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]

                # Now that we've seen both classes, update generator
                generator_optimizer.step()
            

            if use_transformation:
                # PHASE 2: Backprop through other decoder (transformation)
                generator_optimizer.zero_grad()
                
                # for i in range(len(decoder_classes)):
                # TODO: remove once done debugging!!!
                for i in range(1):
                    # This is dumb with two classes, I know
                    decoder_class = decoder_classes[i]
                    other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]

                    # Run features through other decoder
                    feats = feat_dict[decoder_class]
                    targets_other = targets_dict[other_decoder_class]

                    translated_feats = model.forward_decoder(feats, other_decoder_class)

                    r_loss = reconstruction_loss(translated_feats, targets_other)
                    r_loss.backward(retain_graph=True)
                    
                    decoder_class_losses[decoder_class]["transformation_recon_loss"] += r_loss.data[0]
                
                # Now that we've seen both classes, update generator
                generator_optimizer.step()


            # STEP 2: Adversarial training


            if domain_adversarial:
                # Train just domain_adversary
                domain_adversary_optimizer.zero_grad()
                disc_losses = dict()
                for decoder_class in decoder_classes:
                    feats = feat_dict[decoder_class]

                    latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats)
                    
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

    def test(epoch, loaders, noised=True):
        iterators = {decoder_class: iter(loaders[decoder_class]) for decoder_class in decoder_classes}

        decoder_class_losses = {}
        total_batches = float('inf')
        for decoder_class in decoder_classes:
            total_batches = min(total_batches, len(loaders[decoder_class]))
            decoder_class_losses[decoder_class] = {}

            if use_reconstruction:
                decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
            if use_transformation:
                decoder_class_losses[decoder_class]["transformation_recon_loss"] = 0.0
            if run_mode == "vae":
                # Track KL divergence as well
                decoder_class_losses[decoder_class]["autoencoding_kld"] = 0.0
                if use_transformation:
                    decoder_class_losses[decoder_class]["transformation_kld"] = 0.0
            if domain_adversarial:
                decoder_class_losses[decoder_class]["domain_adversarial_loss"] = 0.0
            elif gan:
                decoder_class_losses[decoder_class]["real_gan_loss"] = 0.0
                decoder_class_losses[decoder_class]["fake_gan_loss"] = 0.0
        
        batches_processed = 0
        class_elements_processed = {decoder_class: 0 for decoder_class in decoder_classes}
    
        # TODO: delete!!!
        total_batches = 1

        while batches_processed < total_batches:
            # Get data for each decoder class
            feat_dict = dict()
            targets_dict = dict()
            element_counts = dict()     # Used to get loss per element, rather than batch, in printed output
            for decoder_class in decoder_classes:
                feats, targets = iterators[decoder_class].next()
                element_counts[decoder_class] = feats.size()[0]

                # Set to volatile so history isn't saved (i.e., not training time)
                if on_gpu:
                    feats = feats.cuda()
                    targets = targets.cuda()
                feats = Variable(feats, volatile=True)
                targets = Variable(targets, volatile=True)

                feat_dict[decoder_class] = feats
                targets_dict[decoder_class] = targets
            
            # Handle noising, if desired
            noised_feat_dict = dict()
            for decoder_class in decoder_classes:
                feats = feat_dict[decoder_class]
                if noise_ratio > 0.0 and noised:
                    # Add noise to signal; randomly drop out % of elements
                    noise_matrix = torch.FloatTensor(np.random.binomial(1, 1.0 - noise_ratio, size=feats.size()).astype(float))
                    
                    # Set to volatile so history isn't saved (i.e., not training time)
                    if on_gpu:
                        noise_matrix = noise_matrix.cuda()
                    noise_matrix = Variable(noise_matrix, requires_grad=False, volatile=True)
                    noised_feats = torch.mul(feats, noise_matrix)
                else:
                    noised_feats = feats
                noised_feat_dict[decoder_class] = noised_feats


            # STEP 1: Autoencoder testing


            # for i in range(len(decoder_classes)):
            for i in range(1):
                # This is dumb with two classes, I know
                decoder_class = decoder_classes[i]
                other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]
                

                if use_reconstruction:
                    # PHASE 1: Backprop through same decoder (denoised reconstruction)


                    noised_feats = noised_feat_dict[decoder_class]
                    targets = targets_dict[decoder_class]

                    recon_batch = model.forward_decoder(noised_feats, decoder_class)

                    r_loss = reconstruction_loss(recon_batch, targets)

                    decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]
            
                if use_transformation:
                    # PHASE 2: Transformation


                    # Run (unnoised) features through other decoder
                    feats = feat_dict[decoder_class]
                    feats_other = feat_dict[other_decoder_class]

                    translated_feats = model.forward_decoder(feats, other_decoder_class)

                    r_loss = reconstruction_loss(translated_feats, feats_other)
                    
                    decoder_class_losses[decoder_class]["transformation_recon_loss"] += r_loss.data[0]


            # STEP 2: Adversarial testing


            if domain_adversarial:
                # Test just domain_adversary
                for decoder_class in decoder_classes:
                    feats = feat_dict[decoder_class]

                    latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats)
                    
                    class_prediction = model.domain_adversary.forward(latent)
                    class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "clean" else torch.FloatTensor(np.ones(class_prediction.size()))
                    
                    # Set to volatile so history isn't saved (i.e., not training time)
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)

                    disc_loss = discriminative_loss(class_prediction, class_truth)
                    domain_adv_loss = -disc_loss
                    decoder_class_losses[decoder_class]["domain_adversarial_loss"] += domain_adv_loss.data[0]
            elif gan:
                # Generative adversarial loss
                # Adversary determines whether output is real data, or TRANSFORMED data
                # Convention for classifier: 1 is True, 0 is Fake

                # Test just discriminators
                for i in range(len(decoder_classes)):
                    # This is dumb with two classes, I know
                    decoder_class = decoder_classes[i]
                    other_decoder_class = decoder_classes[(i + 1) % len(decoder_classes)]

                    feats = feat_dict[decoder_class] 

                    # Create minibatch of transformed examples
                    sim_feats = model.forward_decoder(feats, other_decoder_class)
                    
                    # Real examples
                    class_prediction = model.forward_gan(feats, decoder_class)
                    class_truth = torch.FloatTensor(np.ones((class_prediction.size()[0], 1)))
                    
                    # Set to volatile so history isn't saved (i.e., not training time)
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)

                    real_disc_loss = discriminative_loss(class_prediction, class_truth)
                    real_adv_loss = -real_disc_loss
                    decoder_class_losses[decoder_class]["real_gan_loss"] += real_adv_loss.data[0]                
                    
                    # Fake examples
                    class_prediction = model.forward_gan(sim_feats, other_decoder_class)
                    class_truth = torch.FloatTensor(np.zeros((class_prediction.size()[0], 1)))
                    
                    # Set to volatile so history isn't saved (i.e., not training time)
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    class_truth = Variable(class_truth, volatile=True)

                    fake_disc_loss = discriminative_loss(class_prediction, class_truth)
                    fake_adv_loss = -fake_disc_loss
                    decoder_class_losses[other_decoder_class]["fake_gan_loss"] += fake_adv_loss.data[0]                
            batches_processed += 1

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
                shutil.copyfile(ckpt_path, ckpt_path)
        else:
            torch.save(state_obj, ckpt_path)

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
        # print_loss_dict(train_loss_dict, train_element_counts)
        print_loss_dict(train_loss_dict, {"clean": 1, "dirty": 1})
                
        dev_loss_dict = test(epoch, dev_loaders)
        print("\nEPOCH %d DEV" % epoch, flush=True)
        # print_loss_dict(dev_loss_dict, dev_element_counts)
        print_loss_dict(dev_loss_dict, {"clean": 1, "dirty": 1})

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
    checkpoint = torch.load(ckpt_path, map_location=lambda storage,loc: storage)

    # Set up model state and set to eval mode (i.e. disable batch norm)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Loaded checkpoint; best model ready now.")

    train_loss_dict = test(epoch, training_loaders, noised=False)
    print("\nTRAINING SET", flush=True)
    # print_loss_dict(train_loss_dict, train_element_counts)
    print_loss_dict(train_loss_dict, {"clean": 1, "dirty": 1})

    dev_loss_dict = test(epoch, dev_loaders, noised=False)
    print("\nDEV SET", flush=True)
    # print_loss_dict(dev_loss_dict, dev_element_counts)
    print_loss_dict(dev_loss_dict, {"clean": 1, "dirty": 1})

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

