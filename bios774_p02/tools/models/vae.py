"""(Conditional) VAE — lecture L14.

Gaussian decoder with a learned per-feature log-variance, so the ELBO is a
proper lower bound on log p(x) (usable as an anomaly score) and the balance
between reconstruction and KL does not hinge on an arbitrary MSE weight.

params: latent_dim (16), hidden (512,256), ch (32, conv), beta (1.0),
        kl_warmup_epochs (5)
"""
import math

import torch
import torch.nn as nn

from .base import GenerativeModel
from .nets import ConvDecoder, ConvEncoder, mlp, one_hot, out_activation


class VAE(GenerativeModel):
    family = "vae"
    supports_log_prob = True

    def __init__(self, data_shape, n_classes=0, conditional=False, backbone="mlp", out_act=None,
                 latent_dim=16, hidden=(512, 256), ch=32, beta=1.0, kl_warmup_epochs=5):
        super().__init__(data_shape, n_classes, conditional, backbone, out_act,
                         latent_dim=latent_dim, hidden=list(hidden), ch=ch, beta=beta,
                         kl_warmup_epochs=kl_warmup_epochs)
        self.zd = latent_dim
        k = n_classes if self.conditional else 0
        if backbone == "conv":
            self.enc = ConvEncoder(self.data_shape, 2 * latent_dim, ch=ch, extra_in=k)
            self.dec = ConvDecoder(latent_dim + k, self.data_shape, ch=ch, out_act=out_act)
        else:
            D = self.data_shape[0]
            self.enc = mlp(D + k, 2 * latent_dim, hidden)
            self.dec = nn.Sequential(mlp(latent_dim + k, D, tuple(reversed(hidden))), out_activation(out_act))
        self.logvar_x = nn.Parameter(torch.full(self.data_shape, -2.0))
        self.D = int(torch.tensor(self.data_shape).prod())

    # -- conditioning helpers
    def _enc_in(self, x, y):
        if not self.conditional:
            return x
        oh = one_hot(y, self.n_classes)
        if x.dim() == 4:
            oh = oh[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat([x, oh], 1)

    def _dec_in(self, z, y):
        return torch.cat([z, one_hot(y, self.n_classes)], 1) if self.conditional else z

    def encode(self, x, y):
        mu, logvar = self.enc(self._enc_in(x, y)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)

    def _terms(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        xh = self.dec(self._dec_in(z, y))
        lv = self.logvar_x.clamp(-7, 3)
        nll = 0.5 * (((x - xh) ** 2) / lv.exp() + lv + math.log(2 * math.pi))
        nll = nll.flatten(1).sum(1)
        kl_d = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)       # per latent dim
        return nll, kl_d, mu, xh

    def train_step(self, x, y):
        nll, kl_d, _, xh = self._terms(x, y)
        w = self.params["kl_warmup_epochs"]
        beta = self.params["beta"] * (min(1.0, (self.epoch + 1) / w) if w else 1.0)
        loss = (nll + beta * kl_d.sum(1)).mean() / self.D
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()
        return {"loss_per_dim": loss.item(), "kl": kl_d.sum(1).mean().item(),
                "recon_mse": ((x - xh) ** 2).mean().item(), "beta": beta}

    @torch.no_grad()
    def sample(self, n, y=None, temperature=1.0, add_noise=False, **kw):
        """Decoder mean by default (standard for images). add_noise=True also draws the
        Gaussian observation noise — closer to the real data's spread for tabular data."""
        dev = self.logvar_x.device
        z = torch.randn(n, self.zd, device=dev) * temperature
        if self.conditional and y is None:
            raise ValueError("conditional VAE needs class labels to sample")
        xh = self.dec(self._dec_in(z, y))
        if add_noise:
            xh = xh + torch.randn_like(xh) * (0.5 * self.logvar_x.clamp(-7, 3)).exp()
        return xh

    @torch.no_grad()
    def sample_from_reference(self, x_ref, y_ref, mode="posterior", temperature=1.0, add_noise=False):
        """Generate around REAL reference samples instead of the prior p(z).

        posterior   : z ~ q(z | x_ref) with its sd scaled by `temperature`
        interpolate : z = lam*z_i + (1-lam)*z_j for two random references of the same class
        Useful when the aggregate posterior of a (rare) class does not line up with
        p(z) — prior samples conditioned on that label then look like other classes.
        """
        mu, logvar = self.encode(x_ref, y_ref)
        if mode == "interpolate":
            perm = torch.arange(len(mu), device=mu.device)
            if y_ref is not None:
                for k in torch.unique(y_ref):
                    ik = torch.where(y_ref == k)[0]
                    perm[ik] = ik[torch.randperm(len(ik), device=mu.device)]
            lam = torch.rand(len(mu), 1, device=mu.device)
            z = lam * mu + (1 - lam) * mu[perm]
        else:
            z = mu + temperature * (0.5 * logvar).exp() * torch.randn_like(mu)
        xh = self.dec(self._dec_in(z, y_ref))
        if add_noise:
            xh = xh + torch.randn_like(xh) * (0.5 * self.logvar_x.clamp(-7, 3)).exp()
        return xh

    @torch.no_grad()
    def log_prob(self, x, y=None):
        nll, kl_d, _, _ = self._terms(x, y)
        return -(nll + kl_d.sum(1))                     # ELBO (lower bound on log p(x))

    @torch.no_grad()
    def diagnostics(self, x, y):
        nll, kl_d, mu, xh = self._terms(x, y)
        kl_per_dim = kl_d.mean(0)
        active = int((mu.var(0) > 0.01).sum())          # Burda et al. active-unit criterion
        return {
            "kl_total": float(kl_d.sum(1).mean()),
            "kl_per_latent_dim": [round(float(v), 4) for v in kl_per_dim],
            "active_units": active, "latent_dim": self.zd,
            "posterior_collapse_suspected": bool(active == 0 or float(kl_d.sum(1).mean()) < 0.1),
            "recon_mse": float(((x - xh) ** 2).mean()),
            "mean_learned_obs_sd": float((0.5 * self.logvar_x.clamp(-7, 3)).exp().mean()),
            "elbo_per_dim": float(-(nll + kl_d.sum(1)).mean() / self.D),
        }
