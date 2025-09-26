"""
Author: Hun-gyeom Kim
Date: September, 26, 2025
Email: gnsruatkfkd@gmail.com
Organization: MIHlab
"""

import random
import nni
import os
import numpy as np
import torch
import json
import time

# Base path for experiment logs
FOR_SAVE_PATH = '/data/hun-gyeom/02_WetAMD/Experiment_Log/'


def set_seed(seed: int):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_path() -> str:
    """
    Create a unique directory for the current NNI experiment and trial.
    Returns the full path where logs and outputs can be saved.
    """
    experiment_id = nni.get_experiment_id()
    trial_id = nni.get_trial_id()

    experiment_path = os.path.join(FOR_SAVE_PATH, experiment_id)
    save_path = os.path.join(experiment_path, trial_id)

    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_and_load_params(save_path: str, params: dict) -> dict:
    """
    Save hyperparameters to a JSON file and reload them to ensure isolation.
    Handles potential caching issues by removing any existing file before saving.
    Retries reading the file up to 3 times if necessary.
    """
    params_filename = os.path.join(save_path, "params.json")

    # Remove existing file to avoid caching issues
    if os.path.exists(params_filename):
        os.remove(params_filename)

    # Save parameters
    with open(params_filename, "w") as f:
        json.dump(params, f)

    # Reload parameters to ensure consistency
    retries = 3
    for i in range(retries):
        try:
            with open(params_filename, "r") as f:
                trial_params = json.load(f)
            return trial_params
        except FileNotFoundError:
            print(f"Attempt {i + 1}: Parameters file not found. Retrying...")
            time.sleep(1)
    else:
        raise FileNotFoundError("Parameters file could not be found after retries.")
