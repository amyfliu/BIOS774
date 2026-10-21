"""Denoising diffusion probabilistic model — lecture L17.

epsilon-prediction objective (Ho et al., 2020). Sampling with DDIM (Song et
al., 2021) so the number of sampling steps can be far smaller than T — the
main compute knob on a laptop. Optional class conditioning with
classifier-free guidance (labels dropped with prob p_uncond during training).

params: T (1000), schedule ('cosine'|'linear'), ch (64, conv UNet width),
        hidden (512, mlp width), n_layers (4, mlp), p_uncond (0.1),
        ema_decay (0.999; sampling uses an exponential moving average of the
        weights, which matters a lot when training is short)
sample kwargs: steps (100), guidance (1.0 = plain conditional), eta (0.0)
"""
import copy
import math

import torch
import torch.nn as nn

from .base import GenerativeModel
from .nets import MLPDenoiser, SmallUNet


def make_betas(T, schedule):
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, T)
    s = 0.008                                    # cosine schedule (Nichol & Dhariwal, 2021)
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999)


class DDPM(GenerativeModel):
    family = "ddpm"

    def __init__(self, data_shape, n_classes=0, conditional=False, backbone="mlp", out_act=None,
                 T=1000, schedule="cosine", ch=64, hidden=512, n_layers=4, p_uncond=0.1, ema_decay=0.999):
        super().__init__(data_shape, n_classes, conditional, backbone, out_act,
                         T=T, schedule=schedule, ch=ch, hidden=hidden, n_layers=n_layers,
                         p_uncond=p_uncond, ema_decay=ema_decay)
        k = n_classes if self.conditional else 0
        if backbone == "conv":
            self.net = SmallUNet(self.data_shape, ch=ch, n_classes=k)
        else:
            self.net = MLPDenoiser(self.data_shape[0], hidden=hidden, n_layers=n_layers, n_classes=k)
        self.net_ema = copy.deepcopy(self.net).requires_grad_(False)
        self._n_updates = 0
        betas = make_betas(T, schedule)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, 0))
        self.T = T
        self.clip = {"tanh": (-1.0, 1.0), "sigmoid": (0.0, 1.0)}.get(out_act)

    def configure_optimizers(self, lr):
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def _update_ema(self):
        self._n_updates += 1
        k = float(self._n_updates)
        d = min(self.params["ema_decay"], (1 + k) / (10 + k))
        for pe, p in zip(self.net_ema.parameters(), self.net.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.net_ema.buffers(), self.net.buffers()):
            be.copy_(b)

    def _null(self, n, dev):
        return torch.full((n,), self.n_classes, device=dev, dtype=torch.long)

    def train_step(self, x, y):
        n = x.shape[0]
        t = torch.randint(0, self.T, (n,), device=x.device)
        ab = self.alpha_bar[t].view([n] + [1] * (x.dim() - 1))
        eps = torch.randn_like(x)
        xt = ab.sqrt() * x + (1 - ab).sqrt() * eps
        yc = None
        if self.conditional:
            drop = torch.rand(n, device=x.device) < self.params["p_uncond"]
            yc = torch.where(drop, self._null(n, x.device), y.long())
        per = ((self.net(xt, t, yc) - eps) ** 2).flatten(1).mean(1)
        loss = per.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        self._update_ema()
        return {"loss": loss.item()}

    def _eps(self, x, t, y, guidance):
        net = self.net_ema
        if not self.conditional:
            return net(x, t, None)
        e_c = net(x, t, y.long())
        if guidance == 1.0:
            return e_c
        e_u = net(x, t, self._null(x.shape[0], x.device))
        return e_u + guidance * (e_c - e_u)

    @torch.no_grad()
    def sample(self, n, y=None, steps=100, guidance=1.0, eta=0.0, **kw):
        if self.conditional and y is None:
            raise ValueError("conditional DDPM needs class labels to sample")
        dev = self.alpha_bar.device
        x = torch.randn((n,) + self.data_shape, device=dev)
        ts = torch.linspace(self.T - 1, 0, min(steps, self.T), device=dev).long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            eps = self._eps(x, t.repeat(n), y, guidance)
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if self.clip:
                x0 = x0.clamp(*self.clip)
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def diagnostics(self, x, y):
        """Epsilon-prediction MSE at several noise levels. Expected shape: highest at small t
        (tiny noise is hard to identify), falling toward ~0 at large t. MSE near 1.0 at
        all levels means the denoiser has learned nothing yet."""
        n = x.shape[0]
        out = {}
        for frac in (0.05, 0.25, 0.5, 0.75, 0.95):
            t = torch.full((n,), int(frac * (self.T - 1)), device=x.device)
            ab = self.alpha_bar[t].view([n] + [1] * (x.dim() - 1))
            eps = torch.randn_like(x)
            xt = ab.sqrt() * x + (1 - ab).sqrt() * eps
            yc = y.long() if self.conditional else None
            out[f"eps_mse_t{int(frac * 100)}pct"] = float(((self.net_ema(xt, t, yc) - eps) ** 2).mean())
        return out
