"""RealNVP normalizing flow — lecture L16.

Exact log-likelihood via change of variables, so it doubles as a density /
anomaly scorer. Affine coupling layers (Dinh et al., 2017) with ActNorm.
  * mlp backbone (tabular / flattened): alternating parity masks, MLP s,t-nets
  * conv backbone (images): checkerboard masks, small conv s,t-nets, plus
    uniform dequantization + logit transform; requires the 'scale01' prep.
Optional class conditioning (one-hot fed to the s,t-nets).

params: n_layers (8), hidden (256 mlp | 64 conv channels), scale_clamp (2.0),
        logit_alpha (0.05)
"""
import math

import torch
import torch.nn as nn

from .base import GenerativeModel
from .nets import mlp, one_hot

LOG2PI = math.log(2 * math.pi)


class ActNorm(nn.Module):
    def __init__(self, shape):
        super().__init__()
        pshape = (1, shape[0], 1, 1) if len(shape) == 3 else (1, shape[0])
        self.loc = nn.Parameter(torch.zeros(pshape))
        self.logs = nn.Parameter(torch.zeros(pshape))
        self.register_buffer("inited", torch.tensor(0))
        self.spatial = int(torch.tensor(shape[1:]).prod()) if len(shape) == 3 else 1

    def forward(self, x):
        if not bool(self.inited) and self.training:
            with torch.no_grad():
                dims = [0, 2, 3] if x.dim() == 4 else [0]
                self.loc.data = -x.mean(dims, keepdim=True)
                self.logs.data = -torch.log(x.std(dims, keepdim=True) + 1e-6)
                self.inited.fill_(1)
        y = (x + self.loc) * self.logs.exp()
        return y, self.logs.sum() * self.spatial

    def inverse(self, y):
        return y * (-self.logs).exp() - self.loc


class Coupling(nn.Module):
    def __init__(self, mask, net, clamp):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net, self.clamp = net, clamp

    def _st(self, xm, cond):
        h = xm if cond is None else torch.cat([xm, cond], 1)
        s, t = self.net(h).chunk(2, dim=1)
        s = self.clamp * torch.tanh(s / self.clamp)
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, x, cond):
        s, t = self._st(x * self.mask, cond)
        return x * self.mask + (1 - self.mask) * (x * s.exp() + t), s.flatten(1).sum(1)

    def inverse(self, y, cond):
        s, t = self._st(y * self.mask, cond)
        return y * self.mask + (1 - self.mask) * ((y - t) * (-s).exp())


class ConvST(nn.Module):
    def __init__(self, c_in, c_out, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, h, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(h, h, 1), nn.ReLU(),
            nn.Conv2d(h, h, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(h, c_out, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class RealNVP(GenerativeModel):
    family = "flow"
    supports_log_prob = True

    def __init__(self, data_shape, n_classes=0, conditional=False, backbone="mlp", out_act=None,
                 n_layers=8, hidden=None, scale_clamp=2.0, logit_alpha=0.05):
        hidden = hidden or (64 if backbone == "conv" else 256)
        super().__init__(data_shape, n_classes, conditional, backbone, out_act,
                         n_layers=n_layers, hidden=hidden, scale_clamp=scale_clamp,
                         logit_alpha=logit_alpha)
        k = n_classes if self.conditional else 0
        self.norms, self.couplings = nn.ModuleList(), nn.ModuleList()
        self.D = int(torch.tensor(self.data_shape).prod())
        for i in range(n_layers):
            if backbone == "conv":
                C, H, W = self.data_shape
                cb = ((torch.arange(H)[:, None] + torch.arange(W)[None]) % 2).float()
                mask = (cb if i % 2 == 0 else 1 - cb)[None, None].expand(1, C, H, W).clone()
                net = ConvST(C + k, 2 * C, hidden)
            else:
                Dd = self.data_shape[0]
                mask = ((torch.arange(Dd) + i) % 2).float()[None]
                net = mlp(Dd + k, 2 * Dd, (hidden, hidden))
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            self.norms.append(ActNorm(self.data_shape))
            self.couplings.append(Coupling(mask, net, scale_clamp))

    # -- conditioning
    def _cond(self, y, like):
        if not self.conditional:
            return None
        oh = one_hot(y, self.n_classes)
        if like.dim() == 4:
            oh = oh[:, :, None, None].expand(-1, -1, like.shape[2], like.shape[3])
        return oh

    # -- image pre-transform: [0,1] pixels -> dequantized -> logit space
    def _to_logit(self, x, dequant=True):
        a = self.params["logit_alpha"]
        pix = torch.round(x.clamp(0, 1) * 255)
        u = torch.rand_like(pix) if dequant else torch.full_like(pix, 0.5)
        v = (pix + u) / 256.0
        s = a + (1 - 2 * a) * v
        y = torch.log(s) - torch.log1p(-s)
        ldj = (math.log(1 - 2 * a) - torch.log(s) - torch.log1p(-s)).flatten(1).sum(1)
        return y, ldj

    def _from_logit(self, y):
        a = self.params["logit_alpha"]
        return ((torch.sigmoid(y) - a) / (1 - 2 * a)).clamp(0, 1)

    def _forward(self, x, y, dequant=True):
        ldj = torch.zeros(x.shape[0], device=x.device)
        if self.backbone == "conv":
            x, l0 = self._to_logit(x, dequant)
            ldj = ldj + l0
        cond = self._cond(y, x)
        for norm, cp in zip(self.norms, self.couplings):
            x, l1 = norm(x)
            x, l2 = cp(x, cond)
            ldj = ldj + l1 + l2
        return x, ldj

    def _log_density(self, x, y, dequant=True):
        z, ldj = self._forward(x, y, dequant)
        logpz = -0.5 * (z ** 2 + LOG2PI).flatten(1).sum(1)
        lp = logpz + ldj
        if self.backbone == "conv":
            lp = lp - self.D * math.log(256)       # density on [0,1] -> discrete 8-bit pixels
        return lp, z

    def train_step(self, x, y):
        lp, _ = self._log_density(x, y)
        loss = -lp.mean() / self.D
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.opt.step()
        out = {"nll_per_dim": loss.item()}
        if self.backbone == "conv":
            out["bits_per_dim"] = loss.item() / math.log(2)
        return out

    @torch.no_grad()
    def sample(self, n, y=None, temperature=1.0, **kw):
        dev = self.norms[0].loc.device
        z = torch.randn((n,) + self.data_shape, device=dev) * temperature
        cond = self._cond(y, z) if self.conditional else None
        if self.conditional and y is None:
            raise ValueError("conditional flow needs class labels to sample")
        for norm, cp in zip(reversed(self.norms), reversed(self.couplings)):
            z = norm.inverse(cp.inverse(z, cond))
        return self._from_logit(z) if self.backbone == "conv" else z

    @torch.no_grad()
    def log_prob(self, x, y=None):
        return self._log_density(x, y, dequant=False)[0]

    @torch.no_grad()
    def diagnostics(self, x, y):
        lp, z = self._log_density(x, y, dequant=False)
        out = {"nll_per_dim": float(-lp.mean() / self.D),
               "latent_mean": float(z.mean()), "latent_sd": float(z.std()),
               "latent_looks_gaussian": bool(abs(float(z.mean())) < 0.2 and 0.7 < float(z.std()) < 1.3)}
        if self.backbone == "conv":
            out["bits_per_dim"] = out["nll_per_dim"] / math.log(2)
        return out
