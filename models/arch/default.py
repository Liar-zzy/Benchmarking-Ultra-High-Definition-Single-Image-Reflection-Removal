# Define network components here
from matplotlib.colors import Normalize
import torch
from torch import nn
import torch.nn.functional as F
from models.network_uhderrnet import SwinIR as swinir

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     
# in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True
class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        # TODO：train
        
        # TODO：test
        # self.device  = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # self.device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pyramid_module = None
        # TODO：train
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, 240, kernel_size=3, stride=2, norm=norm, act=act)
        # TODO：test
        # self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act).to(self.device1)
        # self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act).to(self.device1)
        # self.conv3 = ConvLayer(conv, n_feats, 240, kernel_size=3, stride=2, norm=norm, act=act).to(self.device1)
        
        
        # TODO:
        # Residual layers
        # dilation_config = [1] * n_resblocks
        # self.res_module = nn.Sequential(*[ResidualBlock(
        #     n_feats, dilation=dilation_config[i], norm=norm, act=act, 
        #     se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])
        
        # TODO：train
        self.swinir_block = swinir(img_size=256, patch_size=4, in_chans=3, embed_dim=240, depths=[6, 6, 6,6,6], num_heads=[6, 6, 6,6,6], window_size=8)
        
        # Upsampling Layers
        # TODO：train
        self.deconv1 = ConvLayer(deconv, 240, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        # TODO：test
        # self.deconv1 = ConvLayer(deconv, 240, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act).to(self.device1)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            # TODO：train
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            #pyramidpooling
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
            
    def forward(self, x):
        # TODO：train
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # TODO:
        # x = self.res_module(x)
        x = self.swinir_block(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        # TODO：test
        # x = self.conv1(x.to(self.device1))
        # x = self.conv2(x.to(self.device1))
        # x = self.conv3(x.to(self.device1))

        # # TODO:
        # # x = self.res_module(x)
        # x = self.swinir_block(x.to(self.device1))
        # x = self.deconv1(x.to(self.device1))
        # x = self.deconv2(x.to(self.device1))
        # if self.pyramid_module is not None:
        #     x = self.pyramid_module(x.to(self.device1))
        # x = self.deconv3(x.to(self.device1))

        return x


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
