import wandb
from types import SimpleNamespace
import os
from copy import deepcopy
from environment.utils import namespace_to_dictionary


def init_wandb(config):
    if config.logger.use_wandb:
        config_out = deepcopy(config)
        config_out = namespace_to_dictionary(config_out)
        
        wandb.login(key='')
        run = wandb.init(
            project=config.logger.project,
            group=config.experiment_name,
            name=config.run_name,
            id=config.run_id,
            config=config_out,
            resume='allow'
        )
    else:
        run = SimpleNamespace()
        run.name = config.run_name
    return run
