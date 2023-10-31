
import torch
import torch.nn as nn

from . import vgg16
from . import unit_vae




def build_model(model_name, num_classes=10, style_layers=None, content_layers=None, device='cpu', conditioned=False, expanded=False, embedding_size=100, alpha=2.0, beta=2.0, layers=None, UseDConv=False, Upsampling_type='nearest'):

    if model_name == 'vgg16':

        model = vgg16.vgg16(style_layers, content_layers)

    elif model_name == 'resnet101':

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights='ResNet101_Weights.IMAGENET1K_V1')
        model.fc = nn.Linear(2048, num_classes)

    elif model_name == 'UNIT':

        model = unit_vae.UNIT_VAE(device=device, conditioned=conditioned, alpha=alpha, beta=beta, UseDConv=UseDConv, Upsampling_type=Upsampling_type)

    elif model_name == 'DED':
        
        model = unit_vae.DEnc_Dec(device=device, UseDConv=UseDConv, Upsampling_type=Upsampling_type)

    return model
