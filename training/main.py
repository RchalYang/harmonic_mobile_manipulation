#!/usr/bin/env python3
"""Entry point to training/validating/testing for a user given experiment
name."""
import os, platform

import hydra
import prior
from omegaconf import DictConfig, OmegaConf

from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from allenact.utils.system import init_logging
from allenact.main import load_config


@hydra.main(config_path=os.path.join(os.getcwd(), "config"), config_name="main")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    import json
    print(json.dumps(OmegaConf.to_container(cfg), indent=2))
    os.makedirs(os.path.expanduser("~/.hydra"), exist_ok=True)

    # NOTE: Support loading in model from prior
    allenact_checkpoint = None
    if cfg.checkpoint is not None and cfg.pretrained_model.name is not None:
        raise ValueError(
            f"Cannot specify both checkpoint {cfg.checkpoint}"
            f" and prior_checkpoint {cfg.pretrained_model.name}"
        )
    elif cfg.checkpoint is None and cfg.pretrained_model.name is not None:
        cfg.checkpoint = prior.load_model(
            project=cfg.pretrained_model.project, model=cfg.pretrained_model.name
        )
        if cfg.eval or not cfg.pretrained_model.only_load_model_state_dict:
            allenact_checkpoint = cfg.checkpoint
    elif cfg.checkpoint is not None:
        if cfg.eval or not cfg.pretrained_model.checkpoint_as_pretrained:
            allenact_checkpoint = cfg.checkpoint

    with open(os.path.expanduser("~/.hydra/config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    os.environ["TRANSFORMERS_OFFLINE"] = cfg.transformers.offline

    if platform.system() != "Darwin":
        init_logging("warning")

    exp_cfg, srcs = load_config(cfg)
    runner = OnPolicyRunner(
        config=exp_cfg,
        output_dir=cfg.output_dir,
        loaded_config_src_files=srcs,
        seed=cfg.seed,
        disable_tensorboard=cfg.disable_tensorboard,
        callbacks_paths=cfg.callbacks,
        mode="test" if cfg.eval else "train",
        machine_id=cfg.distributed.machine_id,
        distributed_ip_and_port=cfg.distributed.ip_and_port,
        distributed_preemption_threshold=cfg.phone2proc.distributed_preemption_threshold,
    )
    if cfg.eval:
        runner.start_test(checkpoint_path_dir_or_pattern=allenact_checkpoint)
    else:
        runner.start_train(
            checkpoint=allenact_checkpoint,
            valid_on_initial_weights=cfg.valid_on_initial_weights,
        )


if __name__ == "__main__":
    main()
