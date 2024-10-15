import os

from omegaconf import OmegaConf

os.makedirs(os.path.expanduser("~/.hydra"), exist_ok=True)
with open(os.path.expanduser("~/.hydra/config.yaml"), "r") as f:
    cfg = OmegaConf.load(f.name)
