"""(Conditional) Wasserstein GAN with gradient penalty — lecture L15.

WGAN-GP rather than the original JS-divergence GAN because the critic gives a
useful training signal even when real and generated distributions barely
overlap (the failure mode discussed in L15). Conditioning: one-hot label
concatenated to z in the generator; projection term in the critic.

params: latent_dim (64), hidden (256,256), ch (32, conv), n_critic (5),
        gp_lambda (10.0)
Each train_step() is one critic update; every n_critic-th call also updates G.
"""
import torch
import torch.nn as nn

from .base import GenerativeModel
from .nets import ConvDecoder, ConvEncoder, mlp, one_hot, out_activation


class WGAN(GenerativeModel):
    family = "wgan"

    def __init__(self, data_shape, n_classes=0, conditional=False, backbone="mlp", out_act=None,
                 latent_dim=64, hidden=(256, 256), ch=32, n_critic=5, gp_lambda=10.0):
        super().__init__(data_shape, n_classes, conditional, backbone, out_act,
                         latent_dim=latent_dim, hidden=list(hidden), ch=ch,
                         n_critic=n_critic, gp_lambda=gp_lambda)
        self.zd = latent_dim
        k = n_classes if self.conditional else 0
        feat = 4 * ch if backbone == "conv" else hidden[-1]
        if backbone == "conv":
            self.G = ConvDecoder(latent_dim + k, self.data_shape, ch=ch, out_act=out_act)
            self.D_feat = ConvEncoder(self.data_shape, feat, ch=ch, norm=False)
        else:
            D = self.data_shape[0]
            self.G = nn.Sequential(mlp(latent_dim + k, D, hidden), out_activation(out_act))
            self.D_feat = mlp(D, feat, hidden, act=lambda: nn.LeakyReLU(0.2))
        self.D_out = nn.Linear(feat, 1)
        self.D_emb = nn.Embedding(n_classes, feat) if self.conditional else None
        self._step = 0

    def configure_optimizers(self, lr):
        dparams = list(self.D_feat.parameters()) + list(self.D_out.parameters())
        if self.D_emb is not None:
            dparams += list(self.D_emb.parameters())
        self.opt_d = torch.optim.Adam(dparams, lr=lr, betas=(0.0, 0.9))
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.9))

    def critic(self, x, y):
        h = torch.nn.functional.leaky_relu(self.D_feat(x), 0.2)
        out = self.D_out(h).squeeze(1)
        if self.D_emb is not None:
            out = out + (self.D_emb(y.long()) * h).sum(1)
        return out

    def generate(self, n, y):
        z = torch.randn(n, self.zd, device=self.D_out.weight.device)
        if self.conditional:
            z = torch.cat([z, one_hot(y, self.n_classes)], 1)
        return self.G(z)

    def train_step(self, x, y):
        n = x.shape[0]
        fake = self.generate(n, y).detach()
        eps = torch.rand([n] + [1] * (x.dim() - 1), device=x.device)
        xi = (eps * x + (1 - eps) * fake).requires_grad_(True)
        grad = torch.autograd.grad(self.critic(xi, y).sum(), xi, create_graph=True)[0]
        gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
        d_real, d_fake = self.critic(x, y).mean(), self.critic(fake, y).mean()
        loss_d = d_fake - d_real + self.params["gp_lambda"] * gp
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()
        out = {"critic_loss": loss_d.item(), "wasserstein_est": (d_real - d_fake).item(), "gp": gp.item()}
        self._step += 1
        if self._step % self.params["n_critic"] == 0:
            loss_g = -self.critic(self.generate(n, y), y).mean()
            self.opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            self.opt_g.step()
            out["gen_loss"] = loss_g.item()
        return out

    @torch.no_grad()
    def sample(self, n, y=None, **kw):
        if self.conditional and y is None:
            raise ValueError("conditional GAN needs class labels to sample")
        return self.generate(n, y)

    @torch.no_grad()
    def diagnostics(self, x, y):
        fake = self.generate(x.shape[0], y)
        f, r = fake.flatten(1), x.flatten(1)
        # Mode-collapse check: spread of generated samples relative to real ones.
        div_fake = torch.pdist(f[:256]).mean().item()
        div_real = torch.pdist(r[:256]).mean().item()
        return {
            "wasserstein_est": float(self.critic(x, y).mean() - self.critic(fake, y).mean()),
            "pairwise_dist_fake": div_fake, "pairwise_dist_real": div_real,
            "diversity_ratio": div_fake / max(div_real, 1e-8),
            "mode_collapse_suspected": bool(div_fake / max(div_real, 1e-8) < 0.5),
        }
