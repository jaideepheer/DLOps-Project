from datetime import datetime
import os
from pathlib import Path

abs_path = os.path.dirname(__file__)
now = datetime.now()
now = now.strftime("%d_%m_%Y")


def build_args(*_, task_name=None, model_name=None, root_dir="/workspace", **overrides):
    assert len(_) == 0
    task_name = task_name or "SOCOFing"
    model_name = model_name or "DeFraudNet"
    root_dir = Path(root_dir or "/workspace").resolve()
    return {
        "data_dir": f"{str(root_dir)}/data/{task_name}",
        "image_dir": f"{str(root_dir)}/data/{task_name}",
        "df_train": f"{str(root_dir)}/data/{task_name}/train.csv",
        "df_val": f"{str(root_dir)}/data/{task_name}/val_split.csv",
        "df_test": f"{str(root_dir)}/data/{task_name}/test.csv",
        "vocab": f"{str(root_dir)}/data/{task_name}/labels.json",
        "triton_dir": f"{str(root_dir)}/triton_model_repository/{task_name}/",
        "GPU_ID": "0",
        "batch_size": 256,
        "num_workers": 16,
        "pin_memory": True,
        "ds_in_memory_cache": True,
        "channels_last": False,
        "epochs": 5,
        "test_size": 0.2,
        "learning_rate": 0.0001,
        ######### MODEL parameters ########
        "model_name": model_name,
        "out_weight_dir": f"{str(root_dir)}/saved_models/{task_name}",
        "out_weight": f"{str(root_dir)}/saved_models/{task_name}/{model_name}.pt",
        "image_size": (96, 103),
        "n_classes": 2,
        ##################################
        "distributed": False,
        "local_rank": 0,
        "amp": False,
        "opt_level": "O2",
        "wandb": True,
        "project_name": f"task_{task_name}",
        "device": "cuda",
        "seed": 42,
        "world_size": 4,
        "TF32": True,
        "benchmark": True,
        **overrides,
    }
