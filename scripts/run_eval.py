import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
from omegaconf import DictConfig, OmegaConf
import pandas as pd
sys.path.append('llm-jp-eval/src') 
sys.path.append('FastChat')
from llm_jp_eval.evaluator import evaluate
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu

# Configuration loading
config_paths = []

config_file = sys.argv[1] if len(sys.argv) > 1 else None

if config_file:
    # Add direct file
    config_paths.append(config_file)

    # Look for yaml/yml
    if not config_file.endswith(('.yaml', '.yml')):
        config_paths.append(f"{config_file}.yaml")
        config_paths.append(f"{config_file}.yml")
       
    # look in configs dir
    config_paths.append(f"configs/{config_file}")
    if not config_file.endswith(('.yaml', '.yml')):
        config_paths.append(f"configs/{config_file}.yaml")
        config_paths.append(f"configs/{config_file}.yml")

    # Default config path
    config_paths.append("configs/config.yaml")

# Try to find and load the config file
cfg_dict = None
for path in config_paths:
    if os.path.exists(path):
        cfg = OmegaConf.load(path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        break

# No configs, let's exit...
if not cfg_dict:
    print("Either pass a YAML config file or have a default `configs/config.yaml` - We won't run without one.")
    sys.exit()


# W&B setup and artifact handling
wandb.login()
run = wandb.init(
    entity=cfg_dict['wandb']['entity'],
    project=cfg_dict['wandb']['project'],
    name=cfg_dict['wandb']['run_name'],
    config=cfg_dict,
    job_type="evaluation",
)

# Initialize the WandbConfigSingleton
WandbConfigSingleton.initialize(run, wandb.Table(dataframe=pd.DataFrame()))
cfg = WandbConfigSingleton.get_instance().config

# Save configuration as artifact
if cfg.wandb.log:
    if os.path.exists("configs/config.yaml"):
        artifact_config_path = "configs/config.yaml"
    else:
        # If "configs/config.yaml" does not exist, write the contents of run.config as a YAML configuration string
        instance = WandbConfigSingleton.get_instance()
        assert isinstance(instance.config, DictConfig), "instance.config must be a DictConfig"
        with open("configs/config.yaml", 'w') as f:
            f.write(OmegaConf.to_yaml(instance.config))
        artifact_config_path = "configs/config.yaml"

    artifact = wandb.Artifact('config', type='config')
    artifact.add_file(artifact_config_path)
    run.log_artifact(artifact)

# Evaluation phase
# 1. llm-jp-eval evaluation
evaluate()
cleanup_gpu()

# 2. mt-bench evaluation
# mtbench_evaluate()
# cleanup_gpu()

# Logging results to W&B
if cfg.wandb.log and run is not None:
    instance = WandbConfigSingleton.get_instance()
    run.log({
        "leaderboard_table": instance.table
    })
    run.finish()
