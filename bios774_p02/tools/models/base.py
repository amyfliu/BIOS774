"""Common interface every generative model implements.

train.py / sample.py / score.py only talk to this interface, so adding a new
family = one new module + one line in models/__init__.py.
"""
import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    family = "base"
    supports_log_prob = False   # exact or bounded per-sample log-likelihood available?

    def __init__(self, data_shape, n_classes=0, conditional=False, backbone="mlp", out_act=None, **params):
        super().__init__()
        self.data_shape = tuple(data_shape)          # (D,) tabular/flat  or (C,H,W) conv
        self.n_classes = int(n_classes)
        self.conditional = bool(conditional) and n_classes > 0
        self.backbone = backbone
        self.out_act = out_act
        self.params = params
        self.epoch = 0

    # -- constructor args, stored in the checkpoint so the model can be rebuilt
    def config(self):
        return dict(data_shape=list(self.data_shape), n_classes=self.n_classes,
                    conditional=self.conditional, backbone=self.backbone,
                    out_act=self.out_act, **self.params)

    def configure_optimizers(self, lr):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def train_step(self, x, y):           # -> dict of floats
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n, y=None, **kw):    # -> tensor in model space
        raise NotImplementedError

    @torch.no_grad()
    def log_prob(self, x, y=None):        # -> per-sample log-likelihood (or bound)
        raise NotImplementedError(f"{self.family} has no tractable per-sample likelihood")

    @torch.no_grad()
    def diagnostics(self, x, y):          # -> dict of family-specific health checks
        return {}

    def _y(self, y):
        return y if self.conditional else None
