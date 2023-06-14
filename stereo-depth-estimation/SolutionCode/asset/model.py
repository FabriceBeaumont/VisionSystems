import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md

# Custom Model
## Unet Feature Fusion
# Here we are adapting the source code of the Unet from https://github.com/qubvel/segmentation_models.pytorch
# to fit our Feature Fusion needs. Specifically, the Unet building blocks are inspired by the ones at: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/decoder.py
# The reason we adapt this code right here instead of writing it from scratch is because the pretrained segmentation backbones available at the repository above are then completely compatible with our Feature Fusion.
# This yields a huge variety of possible encoder backbones that we could use.
### Unet decoder block - Attention is not used
class CustomDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            upsample=True
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

### Unet center block - Conv Layers at the very bottom of the U
### Optional, not used in our final model version
class CustomCenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

### Unet decoder - A stack of several decoder blocks
### This module was modified to provide all intermediate feature maps for fusion
class CustomUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        n_blocks = len(decoder_channels)

        encoder_channels = encoder_channels[:n_blocks+1]
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        upsample = [True for channel in encoder_channels]
        upsample[-2:] = [False, False]
        self.num_downsample = [0 for channel in encoder_channels]
        self.num_downsample[-2:] = [1, 2]
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CustomCenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            CustomDecoderBlock(in_ch, skip_ch, out_ch, upsample=ups, **kwargs)
            for in_ch, skip_ch, out_ch, ups in zip(in_channels, skip_channels, out_channels, upsample)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        feature_maps = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                for j in range(self.num_downsample[i]):
                    skip = self.downsample(skip)
            x = decoder_block(x, skip)
            feature_maps.append(x)
        x = feature_maps[0]
        feature_maps = feature_maps[1:]
        for i in range(len(feature_maps)-2):
                x = torch.cat([self.upsample(x), feature_maps[i]], dim=1)
        for i in range(2):
                x = torch.cat([x, feature_maps[-(i+1)]], dim=1)
        return x

## Unet - The full feature fusion for Unet implementation
### The encoder backbone can be selected from many available pretrained models
class UnetFeatures(nn.Module):
    def __init__(self, encoder_depth=3, decoder_channels=(64, 32, 16, 12, 8), backbone='resnet34'):
        super().__init__()
        model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet", encoder_depth=encoder_depth, decoder_channels=decoder_channels[:encoder_depth])
        self.encoder = model.encoder
        self.decoder = CustomUnetDecoder(encoder_channels=model.encoder._out_channels, decoder_channels=decoder_channels[:encoder_depth], center=True)
        #self._freeze_encoder()

    def _freeze_encoder(self):
        self.encoder.requires_grad_(False)
  
    def _unfreeze_encoder(self):
        self.encoder.requires_grad_(True)

    def forward(self, x):
        enc_features = self.encoder(x)
        return self.decoder(*enc_features)


# stacked hourglass - from PSM net paper https://arxiv.org/pdf/1803.08669.pdf
class StackedHourglass(nn.Module):
    def __init__(self, inplanes, norm_layer = None):
        super(StackedHourglass, self).__init__()
        self.C = inplanes

        self.layer_1_2 = nn.Sequential(nn.Conv3d(self.C, self.C//2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C//2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C//2),
                                       nn.ReLU(inplace=True))
        self.layer_2_4 = nn.Sequential(nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C//2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C//2))
        self.layer_4_6 = nn.Sequential(nn.Conv3d(self.C//2, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                       nn.BatchNorm3d(self.C),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C),
                                       nn.ReLU(inplace=True))
        self.layer_6_8 = nn.Sequential(nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                       nn.BatchNorm3d(self.C),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm3d(self.C),
                                       nn.ReLU(inplace=True))
        self.layer_8_9 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                       nn.BatchNorm3d(self.C))
        self.layer_9_10 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C//2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                        nn.BatchNorm3d(self.C//2))
        self.layer_10_12 = nn.Sequential(nn.Conv3d(self.C//2, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm3d(self.C))
        self.layer_12_14 = nn.Sequential(nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True))
        self.layer_14_15 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(self.C))
        self.layer_15_16 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C//2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(self.C//2)) 
        self.layer_16_18 = nn.Sequential(nn.Conv3d(self.C//2, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm3d(self.C))
        self.layer_18_20 = nn.Sequential(nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(self.C, self.C, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm3d(self.C),
                                         nn.ReLU(inplace=True))
        self.layer_20_21 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(self.C))
        self.layer_21_22 = nn.Sequential(nn.ConvTranspose3d(self.C, self.C//2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(self.C//2))
      
        self.dist1 = nn.Sequential(nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(self.C//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(self.C//2, 1, kernel_size=3, padding=1,bias=False))
        self.dist2 = nn.Sequential(nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(self.C//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(self.C//2, 1, kernel_size=3, padding=1,bias=False))
        self.dist3 = nn.Sequential(nn.Conv3d(self.C//2, self.C//2, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(self.C//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(self.C//2, 1, kernel_size=3, padding=1,bias=False))
    
    def forward(self, x):
        # 2x conv3d 
        a2 = self.layer_1_2(x) #-- 64->32
        # 2x conv3d
        a4 = F.relu(self.layer_2_4(a2) + a2, inplace=True) #-- 32->32
        ## hourglass 1 
        # 1x downconv + 1x conv
        a6 = self.layer_4_6(a4) #-- 32->64
        # 1x downconv + 1x conv
        a8 = self.layer_6_8(a6) #-- 64->64
        # 2x convtranspose
        a9 = F.relu(self.layer_8_9(a8)+ a6, inplace=True) #-- 64->64
        a10 = F.relu(self.layer_9_10(a9)+ a4, inplace=True) #-- 64->32
        ## hourglass 2
        # 1x downconv + 1x conv
        a12 = F.relu(self.layer_10_12(a10) + a9, inplace=True) #-- 32->64
        # 1x downconv + 1x conv
        a14 = self.layer_12_14(a12) #-- 64->64
        # 2x convtranspose
        a15 = F.relu(self.layer_14_15(a14)+ a6, inplace=True) #-- 64->64
        a16 = F.relu(self.layer_15_16(a15)+ a4, inplace=True) #-- 64->32
        ## hourglass 3
        # 1x downconv + 1x conv
        a18 = F.relu(self.layer_16_18(a16) + a15, inplace=True) #-- 32->64
        # 1x downconv + 1x conv
        a20 = self.layer_18_20(a18) #-- 64->64
        # 2x convtranspose
        a21 = F.relu(self.layer_20_21(a20)+ a6, inplace=True) #-- 64->64
        a22 = F.relu(self.layer_21_22(a21)+ a4, inplace=True) #-- 64->32

        c1 = self.dist1(a10) #-- 32->1
        c2 = self.dist2(a16) + c1 #-- 32->1
        c3 = self.dist2(a22) + c2 #-- 32->1
        if self.training:
            return (c1, c2, c3)
        else:
            return c3


# squeeze layer
class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        # type: (Tensor) -> Tensor
        return x.squeeze(self.dim)


# disparity expectation layer
class Expectation(nn.Module):
    def __init__(self, max_disp):
        super(Expectation, self).__init__()
        # this is a parameter without grad to ensure model.cuda() behaves as expected
        self.weights = nn.parameter.Parameter(torch.arange(max_disp).float(), requires_grad=False)
    
    def forward(self, x):
        # type: (Tensor) -> Tensor
        return (x.permute(0,2,3,1) @ self.weights).unsqueeze(1)


# group-wise corr as per GWC net: https://github.com/xy-guo/GwcNet/blob/master/models/submodule.py
class GWC(nn.Module):
    def __init__(self, max_disp, num_groups):
        super(GWC, self).__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
    
    def forward(self, left_feats, right_feats):
        # type: (Tensor, Tensor) -> Tensor
        B, C, H_prime, W_prime = left_feats.shape
        channels_per_group = C // self.num_groups
        cost = (left_feats * right_feats).view([B, self.num_groups, channels_per_group, H_prime, W_prime]).mean(dim=2)
        return cost

# cost as per GWC net: https://github.com/xy-guo/GwcNet/blob/master/models/submodule.py
class GWCBuilder(nn.Module):
    def __init__(self, max_disp, num_groups):
        super(GWCBuilder, self).__init__()
        self.gwc = GWC(max_disp, num_groups)
        self.max_disp = max_disp
        self.num_groups = num_groups
    
    def forward(self, left_feats, right_feats):
        # type: (Tensor, Tensor) -> Tensor
        B, C, H_prime, W_prime = left_feats.shape
        cost = torch.zeros(B, self.num_groups, self.max_disp//4, H_prime, W_prime).to(left_feats.device)
        for i in range(self.max_disp//4):
            if i > 0:
                cost[:, :, i, :, i:] = self.gwc(left_feats[:, :, :, i:], right_feats[:, :, :, :-i])
            else:
                cost[:, :, i, :, :] = self.gwc(left_feats, right_feats)
        return cost


# cost volume construction is taken from the lab slides
class CostBuilder(nn.Module):
    def __init__(self, max_disp):
        super(CostBuilder, self).__init__()
        self.max_disp = max_disp
    
    def forward(self, left_feats, right_feats):
        # type: (Tensor, Tensor) -> Tensor
        B, C, H_prime, W_prime = left_feats.shape
        cost = torch.zeros(B, 2*C, self.max_disp//4, H_prime, W_prime).to(left_feats.device)
        for i in range(self.max_disp//4):
            if i==0:
                cost[:,:C,i,:,:] = left_feats
                cost[:,C:,i,:,:] = right_feats
            else:
                cost[:,:C,i,:,i:] = left_feats[:,:,:,i:]
                cost[:,C:,i,:,i:] = right_feats[:,:,:,:-i]
        return cost

# residual blocks are built like shown in https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, norm_layer = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = None if (stride == 1 and inplanes == planes) else nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))

    def forward(self, x):
        # type: (Tensor) -> Tensor
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def conv3d3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3d1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, norm_layer = None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3d3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = None if (stride == 1 and inplanes == planes) else nn.Sequential(conv3d1x1(inplanes, planes, stride), norm_layer(planes))
    
    def forward(self, x):
        # type: (Tensor) -> Tensor
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class CustomModel(nn.Module):
    def __init__(self, in_channels=3, max_disp=192, encoder_depth=3, decoder_channels=(256, 128, 64, 32, 16), backbone='resnet34'): #136
        super(CustomModel, self).__init__()
        # features:
        # in:  B  x  3 x H    x W
        # out: B x  C x H//4 x W//4
        self.features = UnetFeatures(encoder_depth=encoder_depth, decoder_channels=decoder_channels, backbone=backbone)
        C = sum(decoder_channels[:encoder_depth])
        # independent feature refinement
        # concat features
        # in:   B x C     x H//4 x W//4
        # out:  B x C_Cat x H//4 x W//4
        self.left_conv_cat = BasicBlock(inplanes=C, planes=12)
        self.right_conv_cat = BasicBlock(inplanes=C, planes=12)
        # correlation features
        # in:   B x C     x H//4 x W//4
        # out:  B x C_Cor x H//4 x W//4
        self.left_conv_cor = BasicBlock(inplanes=C, planes=160)
        self.right_conv_cor = BasicBlock(inplanes=C, planes=160)
        # concat cost volume
        # in: 2 of B x   C_Cat           x H//4 x W//4
        # out:     B x 2*C_Cat x Disp//4 x H//4 x W//4
        self.build_cost = CostBuilder(max_disp=max_disp)
        # group-wise cost volume
        # in: 2 of B x C_Cor              x H//4 x W//4
        # out:     B x N_Groups x Disp//4 x H//4 x W//4
        self.build_gwc = GWCBuilder(max_disp=max_disp, num_groups=40)
        # 3d convs
        # in:  B x N_Groups + 2*C_Cat x Disp//4 x H//4 x W//4
        # out: B x 1                  x Disp//4 x H//4 x W//4
        C3d = 64
        self.stacked_hourglass = StackedHourglass(C3d)
        # regressor
        # in:  B x 1 x Disp//4 x H//4 x W//4
        # out: B x 1           x H    x W
        self.regressor = nn.Sequential(
            nn.Upsample(scale_factor=4),
            Squeeze(dim=1),
            torch.nn.Softmax(dim=1),
            Expectation(max_disp)
            )
        
    def forward(self, img_left, img_right):
        # type: (Tensor, Tensor) -> Tensor or (Tensor, Tensor, Tensor)
        _, _, H, W = img_left.shape
        feat_left, feat_right = self.features(img_left), self.features(img_right)
        feat_left_cor, feat_right_cor = self.left_conv_cor(feat_left), self.right_conv_cor(feat_right)
        feat_left_cat, feat_right_cat = self.left_conv_cat(feat_left), self.right_conv_cat(feat_right)
        cost_gwc = self.build_gwc(feat_left_cor, feat_right_cor)
        cost_cat = self.build_cost(feat_left_cat, feat_right_cat)
        cost_volume = torch.cat([cost_gwc, cost_cat], dim=1)
        if self.training:
            (c1, c2, c3) = self.stacked_hourglass(cost_volume)
            return (self.regressor(c1), self.regressor(c2), self.regressor(c3))
        else:
            c3 = self.stacked_hourglass(cost_volume)
            return self.regressor(c3)
        
# Baseline model
class BaselineModel(nn.Module):
    def __init__(self, in_channels=3, max_disp=192):
        super(BaselineModel, self).__init__()
        # features:
        # in: B  x  3 x H    x W
        # out: B x 32 x H//4 x W//4
        self.features = nn.Sequential(
            BasicBlock(inplanes= 3, planes= 8, stride=1),
            BasicBlock(inplanes= 8, planes= 8, stride=1),
            BasicBlock(inplanes= 8, planes=16, stride=2),
            BasicBlock(inplanes=16, planes=16, stride=1),
            BasicBlock(inplanes=16, planes=16, stride=1),
            BasicBlock(inplanes=16, planes=32, stride=2),
            BasicBlock(inplanes=32, planes=32, stride=1),
            BasicBlock(inplanes=32, planes=32, stride=1)
        )
        # cost volume
        # in: 2 of B x 32           x H//4 x W//4
        # out:     B x 64 x Disp//4 x H//4 x W//4
        self.build_cost = CostBuilder(max_disp=max_disp)
        # 3d convs
        # in:  B x 64 x Disp//4 x H//4 x W//4
        # out: B x  1 x Disp//4 x H//4 x W//4
        self.conv_3d = nn.Sequential(
            BasicBlock3D(inplanes=64, planes=32),
            BasicBlock3D(inplanes=32, planes=16),
            BasicBlock3D(inplanes=16, planes= 8),
            BasicBlock3D(inplanes= 8, planes= 1)
        )
        # regressor
        # in:  B x 1 x Disp//4 x H//4 x W//4
        # out: B x 1           x H    x W
        self.regressor = nn.Sequential(
            #nn.ConvTranspose3d(1, 1, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
            #nn.ConvTranspose3d(1, 1, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
            nn.Upsample(scale_factor=4),
            Squeeze(dim=1),
            torch.nn.Softmax(dim=1),
            Expectation(max_disp)
            )
    
    def forward(self, img_left, img_right):
        # type: (Tensor, Tensor) -> Tensor
        feat_left, feat_right = self.features(img_left), self.features(img_right)
        cost_volume = self.build_cost(feat_left, feat_right)
        disp_features = self.conv_3d(cost_volume)
        return self.regressor(disp_features)