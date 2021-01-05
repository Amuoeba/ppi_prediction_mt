# General imports
import os
from pathlib import Path
import time
# Project specific imports
from torch.utils.data import Dataset
import torch
# Imports from internal libraries
import config_old
from datasets.distogram_sequence_dataset import DistogramSequenceDataset
from datasets.distogram_sequence_dataset import DistogramSequenceDataset, PDBindDataset, PandasMolStructure
# Typing imports
from typing import TYPE_CHECKING
from typing import Type
# if TYPE_CHECKING:
    


# TODO Would it be better to have cacher for individual dataset?
class Cacher:
    def __init__(self,dataset:Type[Dataset],cache_dir):
        self.dataset = dataset
        self.cache_dir = cache_dir        
    
    def cache_it(self):
        t = time.localtime()
        self.creation_date = f"{t.tm_year}{t.tm_mon:>02}{t.tm_mday:>02}_{t.tm_hour:>02}_{t.tm_min:>02}_{t.tm_sec:>02}"
        if not os.path.exists(self.cache_dir):
            os.makedirs(f"{self.cache_dir}_{self.creation_date}", exist_ok=True)
            
        for sample in self.dataset:
            a = 1
    

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    sql_db = PDBindDataset(config.folder_structure_cfg.PDBind_sql)
    samples = sql_db.get_2chain_samples()
    train_set = DistogramSequenceDataset(
        samples, 512, set_type="whole", feature_type="stacked")
    
    a =1
