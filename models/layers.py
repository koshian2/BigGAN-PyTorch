import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d

class SNConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        w = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv = spectral_norm(w, eps=1e-4)
        # init
        nn.init.orthogonal_(w.weight)
        if bias:
            nn.init.zeros_(w.bias)

    def forward(self, inputs):
        return self.conv(inputs)


class SNLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        w = nn.Linear(in_features, out_features, bias=bias)
        self.linear = spectral_norm(w, eps=1e-4)
        # init
        nn.init.orthogonal_(w.weight)
        if bias:
            nn.init.zeros_(w.bias)

    def forward(self, inputs):
        return self.linear(inputs)

## Self attention
class SelfAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv_theta = SNConv2d(dims, dims // 8, kernel_size=1, bias=False)
        self.conv_phi = SNConv2d(dims, dims // 8, kernel_size=1, bias=False)
        self.conv_g = SNConv2d(dims, dims // 2, kernel_size=1, bias=False)
        self.conv_attn = SNConv2d(dims // 2, dims, kernel_size=1, bias=False)
        self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        batch, ch, height, width = inputs.size()
        # theta path
        theta = self.conv_theta(inputs)
        theta = theta.view(batch, ch // 8, height * width).permute([0, 2, 1])  # (B, HW, C/8)        
        # phi path
        phi = self.conv_phi(inputs)
        phi = F.max_pool2d(phi, kernel_size=2)  # (B, C/8, H/2, W/2)
        phi = phi.view(batch, ch // 8, height * width // 4)  # (B, C/8, HW/4)
        # attention
        attn = torch.bmm(theta, phi)  # (B, HW, HW/4)
        attn = F.softmax(attn, dim=-1)
        # g path
        g = self.conv_g(inputs)
        g = F.max_pool2d(g, kernel_size=2)  # (B, C/2, H/2, W/2)
        g = g.view(batch, ch // 2, height * width // 4).permute([0, 2, 1])  # (B, HW/4, C/2)

        attn_g = torch.bmm(attn, g)  # (B, HW, C/2)
        attn_g = attn_g.permute([0, 2, 1]).view(batch, ch // 2, height, width)  # (B, C/2, H, W)
        attn_g = self.conv_attn(attn_g)
        return inputs + self.sigma_ratio * attn_g


## BigGAN-deep version of conditional batch norm
## Shared embeddings, sync batch norm
class ConditionalBatchNorm(nn.Module):
    def __init__(self, out_ch, embedding_dims):
        super().__init__()
        # Shared embedding is calculated in initial layer of G 
        # onehot(y) -> shared embedding -> projected
        # [projected + latent z] -> linear(gain, bias) -> conditional batch norm
        # embedding_dims = [projected_dim + latent_z_dim]
        self.out_ch = out_ch
        self.bn = SynchronizedBatchNorm2d(out_ch, affine=False)
        # gain, bias
        self.gain = nn.Linear(embedding_dims, out_ch, bias=False)
        self.bias = nn.Linear(embedding_dims, out_ch, bias=False)
        # init
        nn.init.orthogonal_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)


    def forward(self, x, latent_embedding):
        x_bn = self.bn(x)
        gamma = self.gain(latent_embedding) + 1
        beta = self.bias(latent_embedding)
        out = gamma.view(-1, self.out_ch, 1, 1) * x_bn + beta.view(-1, self.out_ch, 1, 1)
        return out

        
class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        self.linear = SNLinear(n_classes, out_dims, bias=False)

    def forward(self, base_features, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(base_features * wy, dim=1, keepdim=True)
        return output_logits + weighted