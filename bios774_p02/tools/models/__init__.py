"""Registry of generative model families. Add a family = add a module + one line here."""
from .ddpm import DDPM
from .flow import RealNVP
from .gan import WGAN
from .vae import VAE

REGISTRY = {"vae": VAE, "wgan": WGAN, "flow": RealNVP, "ddpm": DDPM}


def build_model(family, **cfg):
    if family not in REGISTRY:
        raise ValueError(f"Unknown model '{family}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[family](**cfg)
