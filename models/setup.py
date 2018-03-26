import os
import sys

import numpy as np

sys.path.append("./")
sys.path.append("./models")
# from pytorch_models import CNNMultidecoder, CNNDomainAdversarialMultidecoder, CNNGANMultidecoder, CNNPhoneMultidecoder, CNNEnd2EndMultidecoder
from pytorch_models import EnhancementNetwork

# Returns the instantiated model based on the current environment variables
def setup_model():
    # Set up features
    feat_dim = int(os.environ["FEAT_DIM"])
    left_context = int(os.environ["LEFT_CONTEXT"])
    right_context = int(os.environ["RIGHT_CONTEXT"])

    freq_dim = feat_dim
    time_dim = (left_context + right_context + 1)

    # Read in parameters to use for our network
    channel_sizes = []
    for res_str in os.environ["CHANNELS_DELIM"].split("_"):
        if len(res_str) > 0:
            channel_sizes.append(int(res_str))
    kernel_sizes = []
    for res_str in os.environ["KERNELS_DELIM"].split("_"):
        if len(res_str) > 0:
            kernel_sizes.append(int(res_str))
    downsample_sizes = []
    for res_str in os.environ["DOWNSAMPLES_DELIM"].split("_"):
        if len(res_str) > 0:
            downsample_sizes.append(int(res_str))
    fc_sizes = []
    for res_str in os.environ["FC_DELIM"].split("_"):
        if len(res_str) > 0:
            fc_sizes.append(int(res_str))
    latent_dim = int(os.environ["LATENT_DIM"])
    
    use_batch_norm = True if os.environ["USE_BATCH_NORM"] == "true" else False
    
       # Construct autoencoder with our parameters
    if domain_adversarial:
        model = CNNDomainAdversarialMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                channel_sizes=channel_sizes,
                                kernel_sizes=kernel_sizes,
                                downsample_sizes=downsample_sizes,
                                fc_sizes=fc_sizes,
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
                                   channel_sizes=channel_sizes,
                                   kernel_sizes=kernel_sizes,
                                   downsample_sizes=downsample_sizes,
                                   fc_sizes=fc_sizes,
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
    elif phone:
        model = CNNPhoneMultidecoder(freq_dim=freq_dim,
                                  splicing=[left_context, right_context], 
                                  channel_sizes=channel_sizes,
                                  kernel_sizes=kernel_sizes,
                                  downsample_sizes=downsample_sizes,
                                  fc_sizes=fc_sizes,
                                  latent_dim=latent_dim,
                                  dec_fc_sizes=dec_fc_sizes,
                                  dec_channel_sizes=dec_channel_sizes,
                                  dec_kernel_sizes=dec_kernel_sizes,
                                  dec_upsample_sizes=dec_upsample_sizes,
                                  activation=activation,
                                  use_batch_norm=use_batch_norm,
                                  decoder_classes=decoder_classes,
                                  weight_init=weight_init,
                                  phone_fc_sizes=phone_fc_sizes,
                                  phone_activation=phone_activation,
                                  num_phones=num_phones)
    elif e2e:
        model = CNNEnd2EndMultidecoder(freq_dim=freq_dim,
                                  splicing=[left_context, right_context], 
                                  channel_sizes=channel_sizes,
                                  kernel_sizes=kernel_sizes,
                                  downsample_sizes=downsample_sizes,
                                  fc_sizes=fc_sizes,
                                  latent_dim=latent_dim,
                                  dec_fc_sizes=dec_fc_sizes,
                                  dec_channel_sizes=dec_channel_sizes,
                                  dec_kernel_sizes=dec_kernel_sizes,
                                  dec_upsample_sizes=dec_upsample_sizes,
                                  activation=activation,
                                  use_batch_norm=use_batch_norm,
                                  decoder_classes=decoder_classes,
                                  weight_init=weight_init,
                                  e2e_fc_sizes=e2e_fc_sizes,
                                  e2e_activation=e2e_activation,
                                  num_phones=num_phones)
    else:
        model = CNNMultidecoder(freq_dim=freq_dim,
                                splicing=[left_context, right_context], 
                                channel_sizes=channel_sizes,
                                kernel_sizes=kernel_sizes,
                                downsample_sizes=downsample_sizes,
                                fc_sizes=fc_sizes,
                                latent_dim=latent_dim,
                                dec_fc_sizes=dec_fc_sizes,
                                dec_channel_sizes=dec_channel_sizes,
                                dec_kernel_sizes=dec_kernel_sizes,
                                dec_upsample_sizes=dec_upsample_sizes,
                                activation=activation,
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
    
    return model



def best_ckpt_path(domain_adversarial=False, gan=False, denoiser=False, phone=False, e2e=False):
    model_dir = os.environ["MODEL_DIR"]

    if domain_adversarial:
        model_name = "best_md_domain"
    elif gan:
        model_name = "best_md_gan"
    elif phone:
        model_name = "best_md_phone"
    elif e2e:
        model_name = "best_md_e2e"
    else:
        model_name = "best_md_vanilla"

    if denoiser:
        model_name += "_denoiser"

    return os.path.join(model_dir, "%s.pth.tar" % model_name)
