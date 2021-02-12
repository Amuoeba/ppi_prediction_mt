# General imports
import os
from pathlib import Path
# Project specific imports

# Imports from internal libraries
import config
import utils
# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:

def get_experiments(exp_root):
    ignore = {".DS_Store", "._.DS_Store"}
    logs = [f"{exp_root}/{x}" for x in os.listdir(exp_root) if x not in ignore]
    return logs


def get_model_types(exp_root):
    experiments = get_experiments(exp_root)
    model_types = set()
    for e in experiments:
        e = Path(e)
        aux_loger = utils.TrainLogger(exp_root, e.name)
        t = aux_loger.get_experiment_type()
        if t:
            model_types.add(t)
    return model_types


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    get_model_types(config.folder_structure_cfg.log_path)
