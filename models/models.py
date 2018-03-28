from collections import OrderedDict
import os
import sys

import torch
from torch import nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch.autograd import Variable



# Convolutional encoder module
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
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
            init_weights(self.weight_init, self.encoder_layers["conv2d_%d" % idx], "conv2d")
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
                # Returns indices in case we have a decoder module with MaxUnpool2d layers
                self.encoder_layers["maxpool2d_%d" % idx] = nn.MaxPool2d((1, downsample_size),
                                                                         return_indices=True)
                
                # Formula from http://pytorch.org/docs/master/nn.html#maxpool2d 
                # Assumes stride = downsample_size (default), padding = 0, dilation = 1
                current_height = current_height     # No change in time dimension!
                current_width = int((current_width - downsample_size) / downsample_size) + 1
        self.encoder = nn.Sequential(self.encoder_layers)

        # STEP 2: Flatten to latent space
        current_fc_dim = current_channels * current_height * current_width
        self.latent_layers = OrderedDict()
        self.latent_layers["lin_final"] = nn.Linear(current_fc_dim, self.latent_dim)
        self.latent = nn.Sequential(self.latent_layers)

    def forward(self, feats):
	# Need to go layer-by-layer to get pooling indices
        pooling_indices = []    
        unpool_sizes = []
        output = feats
        for i, (encoder_layer_name, encoder_layer) in enumerate(self.encoder_layers.items()):
            if "maxpool2d" in encoder_layer_name:
                unpool_sizes.append(output.size())
                output, new_pooling_indices = encoder_layer(output)
                pooling_indices.append(new_pooling_indices)
            else:
                output = encoder_layer(output)
        output_vec = output.view((output.size()[0], -1))
        
        return (self.latent(output_vec), unpool_sizes, pooling_indices)


# De-convolutional decoder module
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Setup initial parameters
        read_base_config(self)
       
        # First, run through encoder setup (without actual layers) to determine end height and width
        current_channels = 1
        current_height = self.time_dim
        current_width = self.freq_dim
        for idx in range(len(self.channel_sizes)):
            current_channels = self.channel_sizes[idx]

            kernel_size = self.kernel_sizes[idx]
            downsample_size = self.downsample_sizes[idx]

            # Formula from http://pytorch.org/docs/master/nn.html#conv2d
            # Assumes stride = 1, dilation = 1, padding = 0
            current_height = current_height - kernel_size + 1
            current_width = current_width - kernel_size + 1
            
            if downsample_size > 0:
                # Formula from http://pytorch.org/docs/master/nn.html#maxpool2d 
                # Assumes stride = downsample_size (default), padding = 0, dilation = 1
                current_height = current_height     # No change in time dimension!
                current_width = int((current_width - downsample_size) / downsample_size) + 1

        # Save values
        self.input_channels = current_channels
        self.input_height = current_height
        self.input_width = current_width

        # Now, actually build decoder

        # First, de-flatten from latent space
        self.latent_layers = OrderedDict()
        self.latent_layers["%s_final" % self.activation] = getattr(nn, self.activation)()
        self.latent_layers["lin_final"] = nn.Linear(self.latent_dim, current_channels * current_height * current_width)
        init_weights(self.weight_init, self.latent_layers["lin_final"], self.activation)
        self.latent = nn.Sequential(self.latent_layers)

        # Now, de-convolve back to original input size
        self.decoder_layers = OrderedDict()
        for idx in reversed(range(len(self.channel_sizes))):
            channel_size = self.channel_sizes[idx]
            kernel_size = self.kernel_sizes[idx]
            downsample_size = self.downsample_sizes[idx]
            
            self.decoder_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()

            if downsample_size > 0:
                # Un-pool only in frequency direction (i.e. kernel and stride 1 in time dimension)
                self.decoder_layers["maxunpool2d_%d" % idx] = nn.MaxUnpool2d((1, downsample_size))
                
                # Formula from http://pytorch.org/docs/master/nn.html#maxunpool2d 
                # Assumes stride = downsample_size (default), padding = 0, dilation = 1
                current_height = current_height     # No change in time dimension!
                current_width = current_width * downsample_size 

            padding = kernel_size - 1
            output_channels = 1 if idx == 0 else self.channel_sizes[idx - 1]

            self.decoder_layers["conv2d_%d" % idx] = nn.Conv2d(channel_size,
                                                               output_channels,
                                                               kernel_size,
                                                               padding=padding)

            # Formula for length from http://pytorch.org/docs/master/nn.html#conv2d
            # Assumes stride = 1, dilation = 1, padding = padding
            current_height = current_height + padding
            current_width = current_width + padding

            if self.use_batch_norm and idx != 0:
                # Don't normalize if it's the output layer!
                self.decoder_layers["batchnorm2d_%d" % idx] = nn.BatchNorm2d(output_channels)

        self.decoder = nn.Sequential(self.decoder_layers)
    
    def forward(self, latent, unpool_sizes, pooling_indices):
        latent_flattened = self.latent(latent)
        latent_deflattened = latent_flattened.view((-1,
                                                    self.input_channels,
                                                    self.input_height,
                                                    self.input_width))

	# Need to go layer-by-layer to insert pooling indices into unpooling layers
        output = latent_deflattened
        for i, (decoder_layer_name, decoder_layer) in enumerate(self.decoder_layers.items()):
            if "maxunpool2d" in decoder_layer_name:
                current_pooling_indices = pooling_indices.pop()
                current_unpool_size = unpool_sizes.pop()
                output = decoder_layer(output,
                                       current_pooling_indices,
                                       output_size=current_unpool_size)
            else:
                output = decoder_layer(output)

        return output
        
        
# Fully-connected phone classifier module
class PhoneClassifier(nn.Module):
    def __init__(self):
        super(PhoneClassifier, self).__init__()

        # Setup initial parameters
        read_base_config(self)

        self.phone_layers = OrderedDict()
        current_fc_dim = self.latent_dim
        for idx in range(len(self.phone_fc_sizes)):
            fc_size = self.phone_fc_sizes[idx]
            
            self.phone_layers["lin_%d" % idx] = nn.Linear(current_fc_dim, fc_size)
            init_weights(self.weight_init, self.phone_layers["lin_%d" % idx], self.activation)
            current_fc_dim = fc_size

            if self.use_batch_norm:
                self.phone_layers["batchnorm1d_%d" % idx] = nn.BatchNorm1d(current_fc_dim)
            self.phone_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()            

        self.phone_layers["lin_final"] = nn.Linear(current_fc_dim, self.num_phones)
        self.phone_layers["LogSoftmax_final"] = nn.LogSoftmax(dim=1)
        self.phone_classifier = nn.Sequential(self.phone_layers)

    def forward(self, latent_vec):
        return self.phone_classifier(latent_vec)


# Acoustic model used either as standalone module or submodule of more complex networks
class AcousticModel(nn.Module):
    def __init__(self, decoder_class="clean"):
        super(AcousticModel, self).__init__()

        self.decoder_class = decoder_class

        # Setup initial parameters
        read_base_config(self)

        self.encoder = Encoder()
        self.phone_classifier = PhoneClassifier()

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_acoustic_model_%s.pth.tar" % self.decoder_class)

    def classify(self, feats):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
		                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        return self.phone_classifier(latent_vec)


# Enhancement network with encoder-decoder tied together
class EnhancementNet(nn.Module):
    def __init__(self):
        super(EnhancementNet, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
        self.encoder = Encoder()
        self.decoder = Decoder()

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_enhancement_net.pth.tar")

    def forward(self, feats):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
                                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        return self.decoder(latent_vec, unpool_sizes, pooling_indices)


# End-to-end phone classifier network with encoder-decoder tied together; no reconstruction loss
class End2EndPhoneNet(nn.Module):
    def __init__(self):
        super(End2EndPhoneNet, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.acoustic_model = AcousticModel()

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_end2end_phone_net.pth.tar")

    def classify(self, feats):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
                                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        enhanced_feats = self.decoder(latent_vec, unpool_sizes, pooling_indices)
        return self.acoustic_model.classify(enhanced_feats)


# End-to-end phone classifier network with encoder-decoder tied together; reconstruction loss
class MultitaskNet(nn.Module):
    def __init__(self):
        super(MultitaskNet, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.acoustic_model = AcousticModel()

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_multitask_net.pth.tar")

    def enhance(self, feats):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
                                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        return self.decoder(latent_vec, unpool_sizes, pooling_indices)

    def classify(self, enhanced_feats):
        return self.acoustic_model.classify(enhanced_feats)

    def forward(self, feats):
        return self.classify(self.enhance(feats))


# Multidecoder used to perform speech enhancement
class EnhancementMultidecoder(nn.Module):
    def __init__(self):
        super(EnhancementMultidecoder, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
        self.encoder = Encoder()
        self.decoder_classes = ["clean", "dirty"]
        self.decoders = dict()
        for decoder_class in self.decoder_classes:
            self.decoders[decoder_class] = Decoder()
            self.add_module("decoder_%s" % decoder_class, self.decoders[decoder_class])

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_enhancement_md.pth.tar")

    def forward_decoder(self, feats, decoder_class):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
                                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        return self.decoders[decoder_class](latent_vec, unpool_sizes, pooling_indices)


# Multidecoder used to perform speech enhancement as well as phone classification
class MultitaskMultidecoder(nn.Module):
    def __init__(self):
        super(MultitaskMultidecoder, self).__init__()

        # Setup initial parameters
        read_base_config(self)
        
        self.encoder = Encoder()
        self.decoder_classes = ["clean", "dirty"]
        self.decoders = dict()
        for decoder_class in self.decoder_classes:
            self.decoders[decoder_class] = Decoder()
            self.add_module("decoder_%s" % decoder_class, self.decoders[decoder_class])
        self.acoustic_model = AcousticModel()

    def ckpt_path(self): 
        return os.path.join(self.model_dir, "best_multitask_md.pth.tar")

    def enhance(self, feats, decoder_class):
        latent_vec, unpool_sizes, pooling_indices = self.encoder(feats.view(-1,
                                                                            1,
                                                                            self.time_dim,
                                                                            self.freq_dim))
        return self.decoders[decoder_class](latent_vec, unpool_sizes, pooling_indices)

    def classify(self, enhanced_feats):
        return self.acoustic_model.classify(enhanced_feats)

    def forward_decoder(self, feats, decoder_class):
        return self.classify(self.enhance(feats, decoder_class))



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

    model.phone_fc_sizes = []
    for res_str in os.environ["PHONE_FC_DELIM"].split("_"):
        if len(res_str) > 0:
            model.phone_fc_sizes.append(int(res_str))

    model.num_phones = int(os.environ["NUM_PHONES"])
    
    # Default parameters
    model.activation = "ReLU"
    model.weight_init = "xavier_uniform"


def init_weights(weight_init, layer, layer_name):
    if "xavier" in weight_init:
        try:
            gain = nn_init.calculate_gain(layer_name.lower())
            getattr(nn_init, weight_init)(layer.weight, gain=gain)
        except ValueError:
            # Happens if layer type isn't supported yet -- just use default gain
            getattr(nn_init, weight_init)(layer.weight)
    else:
        getattr(nn_init, weight_init)(layer.weight)
