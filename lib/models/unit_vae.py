import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None, device='cpu'):
        super(Encoder, self).__init__()

        self.device = device

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = torch.normal(mean = 0.0, std=1.0, size=mu.shape).to(self.device)
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z

class Generator(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None, conditioned=False, alpha=2.0, beta=2.0, UseDConv=False, Upsampling_type='nearest'):
        super(Generator, self).__init__()

        self.conditioned = conditioned
        self.alpha = alpha
        self.beta = beta
        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            if UseDConv:
                #Original Code
                layers += [
                            nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                            nn.InstanceNorm2d(dim // 2),
                            nn.LeakyReLU(0.2, inplace=True),
                          ]
            else:
                #Fixing Checkerboard artifacs
                layers += [
                            nn.Upsample(scale_factor = 2, mode=Upsampling_type),
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(dim, dim // 2,kernel_size=3, stride=1, padding=0),
                            nn.InstanceNorm2d(dim // 2),
                            nn.LeakyReLU(0.2, inplace=True),
                          ]
                #End fixing

            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Sigmoid()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x, y=None):

        if self.conditioned:
            Lambda = torch.from_numpy(np.random.beta(self.alpha, self.beta, size=x.size()[0]))
            Lambda = Lambda.type(torch.FloatTensor)
            Lambda = torch.repeat_interleave(Lambda, np.prod(list(x.size()[1:])))
            Lambda = Lambda.to(x.get_device())
            x = Lambda.view(x.size())*x + (1-Lambda.view(x.size()))*y


        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x

class UNIT_VAE(nn.Module):
    def __init__(self, device='cpu', conditioned=False, alpha=2.0, beta=2.0, UseDConv=False, Upsampling_type='nearest'):
        super(UNIT_VAE, self).__init__()

        shared_dim = 64 * 2 ** 2
        self.conditioned = conditioned
        self.IsEncFrozen = False
        self.IsDecFrozen = False
        self.shared_E = shared_E = ResidualBlock(features=shared_dim)
        self.shared_G = shared_G = ResidualBlock(features=shared_dim)
        self.E = Encoder(in_channels=3, dim=64, n_downsample=2, shared_block=self.shared_E, device=device)
        self.G = Generator(out_channels=3, dim=64, n_upsample=2, shared_block=self.shared_G, conditioned=conditioned, alpha=alpha, beta=alpha, UseDConv=UseDConv, Upsampling_type=Upsampling_type)

    def freezeEncoder(self):
        self.IsEncFrozen = True
        for param in self.E.parameters():
            param.requires_grad = False

    def unfreezeEncoder(self):
        self.IsEncFrozen = False
        for param in self.E.parameters():
            param.requires_grad = True

    def freezeDecoder(self):
        self.IsDecFrozen = True
        for param in self.G.parameters():
            param.requires_grad = False

    def unfreezeDecoder(self):
        self.IsDecFrozen = False
        for param in self.G.parameters():
            param.requires_grad = True

    def forward(self, x, y=None):

        mu, embedding = self.E(x)

        if self.conditioned:
            _, y = self.E(y) #Getting target embedding

        rec_x = self.G(embedding, y)

        return rec_x, mu, embedding

class DEnc_Dec(nn.Module):
    def __init__(self, device='cpu', UseDConv=False, Upsampling_type='nearest'):
        super(DEnc_Dec, self).__init__()

        shared_dim = 64 * 2 ** 2
        self.IsStyleEncFrozen = False
        self.IsContentEncFrozen = False
        self.IsDecFrozen = False

        #Initializing Residual block for the Encoders and Decoder
        self.shared_StyleE = shared_E = ResidualBlock(features=shared_dim)
        self.shared_ContentE = shared_E = ResidualBlock(features=shared_dim)
        self.shared_G = shared_G = ResidualBlock(features=shared_dim)

        #Initializing Encoders and Decoder
        self.StyleE = Encoder(in_channels=3, dim=64, n_downsample=2, shared_block=self.shared_StyleE, device=device)
        self.ContentE = Encoder(in_channels=3, dim=64, n_downsample=2, shared_block=self.shared_ContentE, device=device)
        self.G = Generator(out_channels=3, dim=64, n_upsample=2, shared_block=self.shared_G, UseDConv=UseDConv, Upsampling_type=Upsampling_type)

        self.bottleNeck = nn.Sequential(*[
                                            nn.ReflectionPad2d(3),
                                            nn.Conv2d(shared_dim*2, shared_dim, 7),
                                            nn.InstanceNorm2d(64),
                                            nn.ReLU(inplace=True),
                                        ])


    def freezeStyleEncoder(self):
        self.IsStyleEncFrozen = True
        for param in self.StyleE.parameters():
            param.requires_grad = False

    def freezeContentEncoder(self):
        self.IsContentEncFrozen = True
        for param in self.ContentE.parameters():
            param.requires_grad = False

    def unfreezeStyleEncoder(self):
        self.IsStyleEncFrozen = False
        for param in self.StyleE.parameters():
            param.requires_grad = True

    def unfreezeContentEncoder(self):
        self.IsContentEncFrozen = False
        for param in self.ContentE.parameters():
            param.requires_grad = True

    def freezeDecoder(self):
        self.IsDecFrozen = True
        for param in self.G.parameters():
            param.requires_grad = False

    def unfreezeDecoder(self):
        self.IsDecFrozen = False
        for param in self.G.parameters():
            param.requires_grad = True

    def forward(self, x, y):

        _, s_embedding = self.StyleE(y)
        _, c_embedding = self.ContentE(x)
        embedding = torch.cat((s_embedding,c_embedding), dim=1)
        embedding = self.bottleNeck(embedding)

        rec_x = self.G(embedding)

        return rec_x