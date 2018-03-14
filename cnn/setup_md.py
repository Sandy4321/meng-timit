import os
import sys

import numpy as np

sys.path.append("./")
sys.path.append("./cnn")
from cnn_md import CNNMultidecoder, CNNDomainAdversarialMultidecoder, CNNGANMultidecoder

# Returns the instantiated model based on the current environment variables
def setup_model(domain_adversarial=False, gan=False):
    # Set up features
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

    # Construct autoencoder with our parameters
    print("Constructing model...", flush=True)
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
    
    print("Done constructing model.", flush=True)
    print(model, flush=True)
    
    # Count number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model has %d trainable parameters" % params, flush=True)

    return model



def best_ckpt_path(domain_adversarial=False, gan=False):
    model_dir = os.environ["MODEL_DIR"]
    if domain_adversarial:
        ckpt_path = os.path.join(model_dir, "best_md_domain.pth.tar")
    elif gan:
        ckpt_path = os.path.join(model_dir, "best_md_gan.pth.tar")
    else:
        ckpt_path = os.path.join(model_dir, "best_md_vanilla.pth.tar")
    return ckpt_path
