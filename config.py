# General imports
import os
from dataclasses import dataclass
import pathlib
import dacite
import json
from typing import Optional
import urllib.request
import requests
from tqdm import tqdm
from pathlib import Path

# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:


@dataclass
class SetupConfig:
    # Configurable
    data_root: pathlib.Path
    data_caches: pathlib.Path
    PDBind_data: pathlib.Path
    PDBind_download: pathlib.Path
    PDBind_sql: pathlib.Path
    log_path: pathlib.Path
    pp_root: Optional[pathlib.Path]
    pl_refined_root: Optional[pathlib.Path]
    pl_full_root: Optional[pathlib.Path]
    pp_targz: Optional[pathlib.Path]
    pl_refined_targz: Optional[pathlib.Path]
    pl_full_targz: Optional[pathlib.Path]
    # Static values
    aminoacids_csv: pathlib.Path = os.path.abspath("data/aminoacids.csv")

    def __post_init__(self):
        self.data_caches = self.data_root.joinpath(self.data_caches)
        self.PDBind_data = self.data_root.joinpath(self.PDBind_data)
        self.PDBind_sql = self.data_root.joinpath(self.PDBind_sql)
        self.PDBind_download = self.data_root.joinpath(self.PDBind_download)

    def init_targz_target_roots(self, download_urls: "PPDbindURls"):
        filename = os.path.split(download_urls.protein_protein)[1].replace(".tar.gz", "")
        self.pp_root = self.PDBind_data.joinpath(filename)
        self.pp_targz = self.PDBind_download.joinpath(os.path.split(download_urls.protein_protein)[1])

        filename = os.path.split(download_urls.protein_ligand_refined)[1].replace(".tar.gz", "")
        self.pl_refined_root = self.PDBind_data.joinpath(filename)
        self.pl_refined_targz = self.PDBind_download.joinpath(os.path.split(download_urls.protein_ligand_refined)[1])

        filename = os.path.split(download_urls.protein_ligand_full)[1].replace(".tar.gz", "")
        self.pl_full_root = self.PDBind_data.joinpath(filename)
        self.pl_full_targz = self.PDBind_download.joinpath(os.path.split(download_urls.protein_ligand_full)[1])


@dataclass
class PPDbindURls:
    protein_protein: pathlib.Path = "http://www.pdbbind.org.cn/download/pdbbind_v2019_PP.tar.gz"
    protein_ligand_refined: pathlib.Path = "http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz"
    protein_ligand_full: pathlib.Path = "http://www.pdbbind.org.cn/download/pdbbind_v2019_other_PL.tar.gz"
    year: int = 2019


# Model settings dataclass
@dataclass
class ModelOne:
    DATALOADER_CACHE:pathlib.Path
    BATCH_SIZE: int
    DATALOADER_WORKERS: int
    VIS_WORKERS_ACT: int
    VIS_WORKERS_FILT: int


pdbind_urls = PPDbindURls

converters = {
    pathlib.Path: pathlib.Path
}

cfg_path = "/home/erik/Projects/master_thesis/ppi_prediction_mt/config.json"
with open(cfg_path, "r") as cfg_file:
    raw_config = json.load(cfg_file)

# Initialize folder structure configs
folder_structure_cfg = raw_config["folder_structure"]
folder_structure_cfg = dacite.from_dict(data_class=SetupConfig, data=folder_structure_cfg,
                                        config=dacite.Config(type_hooks=converters))
folder_structure_cfg.init_targz_target_roots(pdbind_urls)

# Initialize model configs
model_one_cfg = raw_config["model_one"]
model_one_cfg = dacite.from_dict(data_class=ModelOne, data=model_one_cfg,
                                 config=dacite.Config(type_hooks=converters))

a = 1
if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    print(folder_structure_cfg.PDBind_data)
