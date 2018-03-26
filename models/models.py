from collections import OrderedDict
import os
import sys

import torch
from torch import nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch.autograd import Variable



# Acoustic model used either as standalone module or submodule of more complex networks
class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        read_acoustic_model_config(self)

        
        # STEP 1: Construct convolutional encoder


        current_channels = 1
        current_height = self.time_dim
        current_width = self.freq_dim

        self.encoder_layers = OrderedDict()
        for idx in range(len(self.channel_sizes)):
            channel_size = self.channel_sizes[idx]
            kernel_size = self.kernel_sizes[idx]
            downsample_size = self.downsample_sizes[idx]

            self.encoder_layers["conv2d_%d" % idx] = nn.Conv2d(current_channels,
                                                               channel_size,
                                                               kernel_size)
            self.init_weights(self.encoder_layers["conv2d_%d" % idx], "conv2d")
            current_channels = channel_size

            # Formula from http://pytorch.org/docs/master/nn.html#conv2d
            # Assumes stride = 1, dilation = 1, padding = 0
            current_height = current_height - kernel_size + 1
            current_width = current_width - kernel_size + 1
            
            if self.use_batch_norm:
                self.encoder_layers["batchnorm2d_%d" % idx] = nn.BatchNorm2d(channel_size)

            self.encoder_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()
            
            if downsample_size > 0:
                # Pool only in frequency direction (i.e. kernel and stride 1 in time dimension)
                self.encoder_layers["maxpool2d_%d" % idx] = nn.MaxPool2d((1, downsample_size))
                
                # Formula from http://pytorch.org/docs/master/nn.html#maxpool2d 
                # Assumes stride = downsample_size (default), padding = 0, dilation = 1
                current_height = current_height     # No change in time dimension!
                current_width = int((current_width - downsample_size) / downsample_size) + 1

        self.encoder = nn.Sequential(self.encoder_layers)
        
        
        # STEP 2: Construct fully-connected phone classifier

        
        self.phone_layers = OrderedDict()
        current_fc_dim = current_channels * current_height * current_width
        for idx in range(len(self.phone_fc_sizes)):
            fc_size = self.phone_fc_sizes[idx]
            
            self.phone_layers["lin_%d" % idx] = nn.Linear(current_fc_dim, fc_size)
            self.init_weights(self.phone_layers["lin_%d" % idx], self.activation)
            current_fc_dim = fc_size

            if self.use_batch_norm:
                self.phone_layers["batchnorm1d_%d" % idx] = nn.BatchNorm1d(current_fc_dim)
            self.phone_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()            

        self.phone_layers["lin_final"] = nn.Linear(current_fc_dim, self.num_phones)
        self.phone_layers["LogSoftmax_final"] = nn.LogSoftmax(dim=1)
        self.phone_classifier = nn.Sequential(self.phone_layers)

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_acoustic_model.pth.tar")

    def init_weights(self, layer, layer_name):
        if "xavier" in self.weight_init:
            try:
                gain = nn_init.calculate_gain(layer_name.lower())
                getattr(nn_init, self.weight_init)(layer.weight, gain=gain)
            except ValueError:
                # Happens if layer type isn't supported yet -- just use default gain
                getattr(nn_init, self.weight_init)(layer.weight)
        else:
            getattr(nn_init, self.weight_init)(layer.weight)

    def classify(self, feats):
        conv_encoded = self.encoder(feats.view(-1,
                                               1,
                                               self.time_dim,
                                               self.freq_dim))
        conv_encoded_vec = conv_encoded.view(conv_encoded.size()[0], -1)
        return self.phone_classifier(conv_encoded_vec)



# Utils for setting up model parameters from environment vars

def read_base_config(model):
    # Read in parameters from base config file
    model.feat_dim = int(os.environ["FEAT_DIM"])
    model.left_context = int(os.environ["LEFT_CONTEXT"])
    model.right_context = int(os.environ["RIGHT_CONTEXT"])

    model.freq_dim = model.feat_dim
    model.time_dim = (model.left_context + model.right_context + 1)

    model.channel_sizes = []
    for res_str in os.environ["CHANNELS_DELIM"].split("_"):
        if len(res_str) > 0:
            model.channel_sizes.append(int(res_str))
    model.kernel_sizes = []
    for res_str in os.environ["KERNELS_DELIM"].split("_"):
        if len(res_str) > 0:
            model.kernel_sizes.append(int(res_str))
    model.downsample_sizes = []
    for res_str in os.environ["DOWNSAMPLES_DELIM"].split("_"):
        if len(res_str) > 0:
            model.downsample_sizes.append(int(res_str))
    model.latent_dim = int(os.environ["LATENT_DIM"])
    model.use_batch_norm = True if os.environ["USE_BATCH_NORM"] == "true" else False

    model.model_dir = os.environ["MODEL_DIR"]

    # Default parameters
    model.activation = "ReLU"
    model.weight_init = "xavier_uniform"

def read_acoustic_model_config(model):
    model.phone_fc_sizes = []
    for res_str in os.environ["PHONE_FC_DELIM"].split("_"):
        if len(res_str) > 0:
            model.phone_fc_sizes.append(int(res_str))

    model.acoustic_model_decoder_classes = []
    for res_str in os.environ["ACOUSTIC_MODEL_DECODER_CLASSES_DELIM"].split("_"):
        if len(res_str) > 0:
            model.acoustic_model_decoder_classes.append(res_str)

    model.num_phones = int(os.environ["NUM_PHONES"])
