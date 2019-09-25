import torch
from torch import nn
import torch.nn.functional as F
from models.layers import SNLinear, SNConv2d, ConditionalBatchNorm, SelfAttention, SNEmbedding
from models.sync_batchnorm import SynchronizedBatchNorm2d

# Multi input version of nn.Sequential
# https://github.com/pytorch/pytorch/issues/19808
class MultiSeqential(nn.Sequential):
    def forward(self, *inputs):
        x, e = inputs
        for module in self._modules.values():
            if type(module) is GBlock:
                x = module(x, e)
            else:
                x = module(x)
        return x

## Generator ##
class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample, embedding_dims, bottleneck_ratio=4):
        super().__init__()
        # conv layers
        hidden_ch = out_ch // bottleneck_ratio
        self.conv1 = SNConv2d(in_ch, hidden_ch, kernel_size=1)
        self.conv2 = SNConv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv3 = SNConv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv4 = SNConv2d(hidden_ch, out_ch, kernel_size=1)
        self.out_ch = out_ch
        # bn layers
        self.bn1 = ConditionalBatchNorm(in_ch, embedding_dims)
        self.bn2 = ConditionalBatchNorm(hidden_ch, embedding_dims)
        self.bn3 = ConditionalBatchNorm(hidden_ch, embedding_dims)
        self.bn4 = ConditionalBatchNorm(hidden_ch, embedding_dims)
        # upsample
        self.upsample = upsample

    def forward(self, x, embedding):
        # skip connection path
        skip = x[:,:self.out_ch,:,:]  # drop channels        
        # main path
        # pre-act
        h = F.relu(self.bn1(x, embedding))
        # 1st conv
        h = F.relu(self.bn2(self.conv1(h), embedding))
        # upsampling if needed
        if self.upsample > 1:
            h = F.interpolate(h, scale_factor=self.upsample)
            skip = F.interpolate(skip, scale_factor=self.upsample)
        # 2nd, 3rd conv
        h = F.relu(self.bn3(self.conv2(h), embedding))
        h = F.relu(self.bn4(self.conv3(h), embedding))
        # last conv
        h = self.conv4(h)
        # add
        return h + skip

def G_arch(ch=64, attention='64'):
    arch = {}
    arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
                'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
                'upsample' : [True] * 6,
                'resolution' : [8, 16, 32, 64, 128, 256],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                for i in range(3,9)}}
    arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
                'out_channels' : [ch * item for item in [16, 8, 4,  2, 1]],
                'upsample' : [True] * 5,
                'resolution' : [8, 16, 32, 64, 128],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                for i in range(3,8)}}
    arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
                'out_channels' : [ch * item for item in [16, 8, 4, 2]],
                'upsample' : [True] * 4,
                'resolution' : [8, 16, 32, 64],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                for i in range(3,7)}}
    arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
                'out_channels' : [ch * item for item in [4, 4, 4]],
                'upsample' : [True] * 3,
                'resolution' : [8, 16, 32],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                for i in range(3, 6)}}
    return arch

### # parameters (Generator) ###
## paper setting = base_ch = 128 
## ImageNet base (n_classes=1000, n_projectd_dims=128)
# --- resolution = 32 ---
# base_ch=16, params=793,190
# base_ch=32, params=1,568,838
# base_ch=48, params=2,455,078
# base_ch=64, params=3,451,910
# base_ch=96, params=5,777,350
# base_ch=128, params=8,545,158
# --- resolution = 64 ---
# base_ch=16, params=2,861,247
# base_ch=32, params=6,598,135
# base_ch=48, params=11,338,799
# base_ch=64, params=17,083,239
# base_ch=96, params=31,583,447
# base_ch=128, params=50,098,759
# --- resolution = 128 ---
# base_ch=16, params=2,914,532
# base_ch=32, params=6,709,632
# base_ch=48, params=11,513,436
# base_ch=64, params=17,325,944
# base_ch=96, params=31,977,072
# base_ch=128, params=50,663,016
# --- resolution = 256 ---
# base_ch=16, params=3,218,085
# base_ch=32, params=7,464,193
# base_ch=48, params=12,866,461
# base_ch=64, params=19,424,889
# base_ch=96, params=36,010,225
# base_ch=128, params=57,220,201

class Generator(nn.Module):
    def __init__(self, base_ch, resolution, n_classes, n_projected_dims=32, n_latent_dims=128):
        super().__init__()
        assert resolution in [32, 64, 128, 256]
        # Onehot vector projection dims [n_classes -> n_projected_dims]
        # Latent random variable dims [n_latent_dims]
        # Conditional batch norm  [n_projected_dims + n_latent_dims -> image path ch]

        # base ch
        self.base_ch = base_ch
        # n_latent_dims random variable of latent space
        self.n_latent_dims = n_latent_dims
        # n_classes of label
        self.n_classes = n_classes
        # n_dimentions projected by sheared embedding
        self.n_projected_dims = n_projected_dims
        embedding_dims = n_projected_dims + n_latent_dims

        # Shared embedding across condtional batch norms
        self.shared_embedding = SNLinear(n_classes, n_projected_dims)
        # architecture of G
        arch = G_arch(base_ch)[resolution]

        # initial linear
        self.initial_ch = arch["in_channels"][0]
        self.linear = SNLinear(embedding_dims, 4 * 4 * self.initial_ch)        

        # main model
        blocks = []
        for in_ch, out_ch, upsample, _, attention in zip(*
                (v.values() if type(v) is dict else v for v in arch.values())):
            # ResBlock with non-upsampling
            blocks.append(
                GBlock(in_ch, in_ch, 1, embedding_dims)                
            )
            # ResBlock with upsampling
            blocks.append(
                GBlock(in_ch, out_ch, 2 if upsample else 1, embedding_dims)                
            )
            # Non-Local block if needed (Self-attention)
            blocks.append(SelfAttention(out_ch))
        self.main = MultiSeqential(*blocks)
            
        # final layers (not to use SNConv or Conditional BN)
        last_ch = arch["out_channels"][-1]
        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(last_ch),
            nn.ReLU(True),
            nn.Conv2d(last_ch, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, y):
        assert z.size(1) == self.n_latent_dims and y.size(1) == self.n_classes
        # linear
        projection = self.shared_embedding(y)
        # shared embedding value (becase BigGAN-deep does not split z)
        embedding = torch.cat([projection, z], dim=1) 
        x = self.linear(embedding).view(z.size(0), self.initial_ch, 4, 4)
        # main convolution blocks
        x = self.main(x, embedding)
        # output conv layers
        x = self.out_conv(x)
        return x

## Discriminator ##
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample, bottleneck_ratio=4):
        super().__init__()
        # conv blocks
        hidden_ch = in_ch // bottleneck_ratio
        self.conv1 = SNConv2d(in_ch, hidden_ch, kernel_size=1)
        self.conv2 = SNConv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv3 = SNConv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv4 = SNConv2d(hidden_ch, out_ch, kernel_size=1)
        # short-conv for increasing channel
        self.downsample = downsample
        if in_ch < out_ch:
            self.conv_short = SNConv2d(in_ch, out_ch - in_ch, kernel_size=1)
        else:
            self.conv_short = None

    def forward(self, inputs):
        # main path
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # avg pool
        if self.downsample > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsample)
            s = F.avg_pool2d(inputs, kernel_size=self.downsample)
        else:
            s = inputs
        x = self.conv4(x)
        # short cut path
        if self.conv_short is not None:
            s_increase = self.conv_short(s)
            s = torch.cat([s, s_increase], dim=1)
        # redisual add
        return x + s

def D_arch(ch=64, attention='64'):
    arch = {}
    arch[256] = {'in_channels' :  [item * ch for item in [1, 2, 4, 8, 8, 16]],
                'out_channels' : [item * ch for item in [2, 4, 8, 8, 16, 16]],
                'downsample' : [True] * 6 + [False],
                'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
                'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                for i in range(2,8)}}
    arch[128] = {'in_channels' :  [item * ch for item in [1, 2, 4,  8, 16]],
                'out_channels' : [item * ch for item in [2, 4, 8, 16, 16]],
                'downsample' : [True] * 5 + [False],
                'resolution' : [64, 32, 16, 8, 4, 4],
                'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                for i in range(2,8)}}
    arch[64]  = {'in_channels' :  [item * ch for item in [1, 2, 4, 8]],
                'out_channels' : [item * ch for item in [2, 4, 8, 16]],
                'downsample' : [True] * 4 + [False],
                'resolution' : [32, 16, 8, 4, 4],
                'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                for i in range(2,7)}}
    arch[32]  = {'in_channels' :  [item * ch for item in [4, 4, 4]],
                'out_channels' : [item * ch for item in [4, 4, 4]],
                'downsample' : [True, True, False, False],
                'resolution' : [16, 16, 16, 16],
                'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                for i in range(2,6)}}
    return arch

### # parameters (Discriminator) ###
## paper setting = base_ch = 128 
## ImageNet base (n_classes=1000)
# --- resolution = 32 ---
# base_ch=16, params=106,465
# base_ch=32, params=292,801
# base_ch=48, params=559,009
# base_ch=64, params=905,089
# base_ch=96, params=1,836,865
# base_ch=128, params=3,088,129
# --- resolution = 64 ---
# base_ch=16, params=462,445
# base_ch=32, params=1,332,889
# base_ch=48, params=2,611,333
# base_ch=64, params=4,297,777
# base_ch=96, params=8,894,665
# base_ch=128, params=15,123,553
# --- resolution = 128 ---
# base_ch=16, params=758,254
# base_ch=32, params=2,514,330
# base_ch=48, params=5,268,230
# base_ch=64, params=9,019,954
# base_ch=96, params=19,516,874
# base_ch=128, params=34,005,090
# --- resolution = 256 ---
# base_ch=16, params=811,950
# base_ch=32, params=2,728,218
# base_ch=48, params=5,748,806
# base_ch=64, params=9,873,714
# base_ch=96, params=21,436,490
# base_ch=128, params=37,416,546

class Discriminator(nn.Module):
    def __init__(self, base_ch, resolution, n_classes):
        super().__init__()
        assert resolution in [32, 64, 128, 256]
        arch = D_arch(base_ch)[resolution]
        # initial conv
        self.initial_conv = SNConv2d(3, arch["in_channels"][0], kernel_size=3, padding=1)
        # main_conv
        blocks = []
        for in_ch, out_ch, downsample, _, attention in zip(*
                (v.values() if type(v) is dict else v for v in arch.values())):
            # D block with downsampling
            blocks.append(DBlock(in_ch, out_ch, 2 if downsample else 1))            
            # D block with non-downsampling
            blocks.append(DBlock(out_ch, out_ch, 1))
            # Non-local(self attention) if needed
            if attention:
                blocks.append(SelfAttention(out_ch))
        self.main = nn.Sequential(*blocks)
        # prob-linear
        self.linear_out = SNLinear(out_ch, 1)        
        # projection
        self.proj_embedding = SNEmbedding(n_classes, out_ch)
        
    def forward(self, x, y):
        h = self.initial_conv(x)
        h = F.relu(self.main(h))
        h = torch.sum(h, dim=(2, 3)) # global sum pooling
        logit = self.linear_out(h)
        h = self.proj_embedding(h, logit, y)
        return h

            
