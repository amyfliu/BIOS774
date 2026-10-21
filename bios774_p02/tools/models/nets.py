"""Small building blocks shared by the generative models (MLP + small CNN backbones)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_out, hidden=(512, 256), act=nn.SiLU, norm=False, final_act=None):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h)]
        if norm:
            layers += [nn.LayerNorm(h)]
        layers += [act()]
        d = h
    layers += [nn.Linear(d, d_out)]
    if final_act is not None:
        layers += [final_act]
    return nn.Sequential(*layers)


def out_activation(name):
    return {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name, nn.Identity())


def one_hot(y, n_classes):
    return F.one_hot(y.long(), n_classes).float()


class ConvEncoder(nn.Module):
    """(C,H,W) -> vector. H, W must be divisible by 4 (28 -> 14 -> 7)."""

    def __init__(self, shape, d_out, ch=32, extra_in=0, norm=True):
        super().__init__()
        C, H, W = shape
        Norm = (lambda c: nn.GroupNorm(8, c)) if norm else (lambda c: nn.Identity())
        self.net = nn.Sequential(
            nn.Conv2d(C + extra_in, ch, 3, 1, 1), Norm(ch), nn.SiLU(),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), Norm(2 * ch), nn.SiLU(),
            nn.Conv2d(2 * ch, 4 * ch, 4, 2, 1), Norm(4 * ch), nn.SiLU(),
        )
        self.fc = nn.Linear(4 * ch * (H // 4) * (W // 4), d_out)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class ConvDecoder(nn.Module):
    """vector -> (C,H,W)."""

    def __init__(self, d_in, shape, ch=32, out_act=None):
        super().__init__()
        C, H, W = shape
        self.h, self.w, self.ch = H // 4, W // 4, ch
        self.fc = nn.Linear(d_in, 4 * ch * self.h * self.w)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(4 * ch, 2 * ch, 3, 1, 1), nn.GroupNorm(8, 2 * ch), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, C, 3, 1, 1),
        )
        self.out = out_activation(out_act)

    def forward(self, z):
        h = self.fc(z).view(-1, 4 * self.ch, self.h, self.w)
        return self.out(self.net(h))


# ----------------------------------------------------------------------------- diffusion backbones
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """Tiny UNet for 28x28-style images: H -> H/2 -> H/4 and back, time + class embeddings."""

    def __init__(self, shape, ch=64, n_classes=0):
        super().__init__()
        C = shape[0]
        e = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, e), nn.SiLU(), nn.Linear(e, e))
        self.cemb = nn.Embedding(n_classes + 1, e) if n_classes else None  # last index = "no label"
        self.inp = nn.Conv2d(C, ch, 3, 1, 1)
        self.d1 = ResBlock(ch, ch, e)
        self.down1 = nn.Conv2d(ch, ch, 4, 2, 1)
        self.d2 = ResBlock(ch, 2 * ch, e)
        self.down2 = nn.Conv2d(2 * ch, 2 * ch, 4, 2, 1)
        self.mid1 = ResBlock(2 * ch, 2 * ch, e)
        self.mid2 = ResBlock(2 * ch, 2 * ch, e)
        self.up2 = nn.Upsample(scale_factor=2)
        self.u2 = ResBlock(4 * ch, ch, e)
        self.up1 = nn.Upsample(scale_factor=2)
        self.u1 = ResBlock(2 * ch, ch, e)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, C, 3, 1, 1))

    def forward(self, x, t, y=None):
        e = self.temb(timestep_embedding(t, self.ch))
        if self.cemb is not None and y is not None:
            e = e + self.cemb(y)
        h0 = self.inp(x)
        h1 = self.d1(h0, e)
        h2 = self.d2(self.down1(h1), e)
        h = self.mid2(self.mid1(self.down2(h2), e), e)
        h = self.u2(torch.cat([self.up2(h), h2], 1), e)
        h = self.u1(torch.cat([self.up1(h), h1], 1), e)
        return self.out(h)


class MLPDenoiser(nn.Module):
    """Denoiser for tabular diffusion: residual MLP with time + class embeddings."""

    def __init__(self, d, hidden=512, n_layers=4, n_classes=0):
        super().__init__()
        self.hidden = hidden
        self.temb = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cemb = nn.Embedding(n_classes + 1, hidden) if n_classes else None
        self.inp = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, d))

    def forward(self, x, t, y=None):
        e = self.temb(timestep_embedding(t, 128))
        if self.cemb is not None and y is not None:
            e = e + self.cemb(y)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h + e)
        return self.out(h)
