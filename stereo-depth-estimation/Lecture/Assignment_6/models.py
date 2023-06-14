import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

#### This is the adapted model from class.
#### The adjustments are straightforward, but the decoder output nonlin was removed
#### This is because we use logits in our BCE term
def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()


class VanillaVAE(nn.Module):
    """
    Implementation of a fully connect autoencoder for 
    denoising  MNIST images
    """
    
    def __init__(self, in_size=(3,32,32), sizes=[3072, 128, 10], act="ReLU"):
        """ Model initlaizer """
        assert np.prod(in_size) == sizes[0]
        super().__init__()
        
        self.in_size = in_size
        self.sizes = sizes
        self.activation = get_act(act) 
        
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1])
        self.fc_sigma = nn.Linear(sizes[-2], sizes[-1])
        #self.log_scale = nn.Parameter(torch.zeros(1))
        return
        
    def _make_encoder(self):
        """ Defining encoder """
        layers = [nn.Flatten()]
        
        # adding fc+act+drop for each layer
        for i in range(len(self.sizes)-2):
            layers.append( nn.Linear(in_features=self.sizes[i], out_features=self.sizes[i+1]) )
            layers.append( self.activation )
                
        # replacing last act and dropout with sigmoid
        encoder = nn.Sequential(*layers)
        return encoder
    
    def _make_decoder(self):
        """ Defining decoder """
        layers = []
        
        # adding fc+act+drop for each layer
        for i in range(1, len(self.sizes)):
            layers.append( nn.Linear(in_features=self.sizes[-i], out_features=self.sizes[-i-1]) )
            layers.append( self.activation )
                
        # replacing last act and dropout with sigmoid
        layers = layers[:-1] #+ [nn.Tanh()] #+ [nn.Sigmoid()]
        decoder = nn.Sequential(*layers)
        return decoder
        
    
    def reparameterize(self, mu, log_var):
        """ Reparametrization trick"""
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)  # random sampling happens here
        z = mu + std * eps
        return z

    
    def forward(self, x):
        """ Forward pass """
        # encoding and computng statistics
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_sigma(x_enc)
        
        # reparametrization trick
        z = self.reparameterize(mu, log_var)
        
        # decoding
        x_hat_flat = self.decoder(z)
        x_hat = x_hat_flat.view(-1, *self.in_size)
        
        return x_hat, (z, mu, log_var)

###This our conv model
###The encoder is taken from a ResNet pretrained on CIFAR10
###The decoder uses residual conv blocks in a somewhat symmetrical manner relative to the encoder
###Upsampling is handled by a choice of: transpose conv; pixelshuffle; upsample followed by conv
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
      
    def forward(self, x):
        return x.view(-1, *self.shape)
    
class Interpolate(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.intp = F.interpolate
      
    def forward(self, x):
        return self.intp(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

class ResConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(channels,eps=1e-05, momentum=0.1, affine=True)
        self.nonLin = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(channels,eps=1e-05, momentum=0.1, affine=True)
      
    def forward(self, x):
        out = self.conv1(x)
        out = self.batchNorm1(out)
        out = self.nonLin(out)

        out = self.conv2(out)
        out = self.batchNorm2(out)

        out += x
        out = self.nonLin(out)

        return out


class ConvVAE(nn.Module):
    def __init__(self, z_dim=64, mode='deConv'):
        super().__init__()
        self.z_dim = z_dim
        self.mode = mode
        
        self.encoder = self.make_encoder()
        if mode == 'deConv':
            self.decoder = self.make_decoder_transpose_conv()
        elif mode == 'pixelShuffle':
            self.decoder = self.make_decoder_pixel_shuffle()
        elif mode == 'upConv':
            self.decoder = self.make_decoder_upsample_conv()

        self.fc_mu = nn.Linear(64*8*8, self.z_dim)
        self.fc_sigma = nn.Linear(64*8*8, self.z_dim)
        #self.log_scale = nn.Parameter(torch.zeros(1))
        return
        
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        p = torch.distributions.normal.Normal(mu, std)
        return p.rsample()

    def make_encoder(self):
        resnet20 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        encoder_list = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        encoder_list +=  [module for (_, module) in resnet20.named_children()]
        encoder_list = encoder_list[:-1]
        encoder_list[-1] = nn.Flatten()
        return nn.Sequential(*encoder_list)
    
    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
    
    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)

    def make_decoder_transpose_conv(self):
        decoder_list =  [nn.Linear(self.z_dim,64*8*8), 
                    Reshape((64,8,8)),
                    nn.ReLU(inplace=True),
                    # upconv + 2x BB
                    nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2, output_padding=1),
                    ResConv(32),
                    ResConv(32),
                    # upconv + 2x BB
                    nn.ConvTranspose2d(32, 16, kernel_size=1, stride=2, output_padding=1),
                    ResConv(16),
                    ResConv(16),
                    # conv + BN + nonLin
                    nn.Conv2d(16,3,3,padding=1),
                    nn.BatchNorm2d(3),
                    nn.ReLU(inplace=True),
                    # conv + outNonLin
                    nn.Conv2d(3,3,3,padding=1)]
        return nn.Sequential(*decoder_list)
    
    def make_decoder_pixel_shuffle(self):
        decoder_list =  [nn.Linear(self.z_dim,64*8*8), 
                 Reshape((64,8,8)),
                 # upconv + 2x BB
                 nn.Conv2d(64, 32*2*2, 1),
                 nn.PixelShuffle(2),
                 ResConv(32),
                 ResConv(32),
                 # upconv + 2x BB
                 nn.Conv2d(32, 16*2*2, 1),
                 nn.PixelShuffle(2),
                 ResConv(16),
                 ResConv(16),
                 # conv + BN + nonLin
                 nn.Conv2d(16,3,3,padding=1),
                 nn.BatchNorm2d(3),
                 nn.ReLU(inplace=True),
                 # conv + outNonLin
                 nn.Conv2d(3,3,3,padding=1)]
        return nn.Sequential(*decoder_list)
    
    def make_decoder_upsample_conv(self):
        decoder_list =  [nn.Linear(self.z_dim,64*8*8), 
                 Reshape((64,8,8)),
                 # upconv + 2x BB
                 Interpolate(2),
                 nn.Conv2d(64,32,3,padding=1),
                 ResConv(32),
                 ResConv(32),
                 # upconv + 2x BB
                 Interpolate(2),
                 nn.Conv2d(32,16,3,padding=1),
                 ResConv(16),
                 ResConv(16),
                 # conv + BN + nonLin
                 nn.Conv2d(16,3,3,padding=1),
                 nn.BatchNorm2d(3),
                 nn.ReLU(inplace=True),
                 # conv + outNonLin
                 nn.Conv2d(3,3,3,padding=1)] ##nn.Tanh()
        return nn.Sequential(*decoder_list)
    
    def forward(self, x):
        """ Forward pass """
        # encoding and computng statistics
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_sigma(x_enc)
        # reparametrization trick
        z = self.reparameterize(mu, log_var)
        
        # decoding
        x_hat = self.decoder(z)
        return x_hat, (z, mu, log_var)