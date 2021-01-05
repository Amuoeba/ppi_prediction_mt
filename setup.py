# General imports
import os
import sys
import pathlib
import argparse
import requests
from tqdm import tqdm
import shutil
import tarfile
import time
# Project specific imports
from config import folder_structure_cfg, pdbind_urls
import mol_readers.pdbind as pdbind_sql
from datasets.distogram_sequence_dataset import DistogramSequenceDataset
import config

from datasets import distogram_sequence_dataset


# Imports from internal libraries


def list_files_in_dir(root_dir):
    files_in_dir = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            f_path = os.path.join(root, f)
            files_in_dir.append(f_path)
    return files_in_dir


def get_arguments():
    arg_parser = argparse.ArgumentParser(description="Set up project environment")
    arg_parser.add_argument("--clean", dest="clean", action="store_true")
    args = arg_parser.parse_args(sys.argv[1:])
    return args


def download(url, fname):
    if not os.path.exists(fname):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file:
            with tqdm(
                    desc=str(fname),
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
    else:
        print(f"File allready exists {fname}")


def extract_targz(filename, target):
    if not os.path.exists(target):
        with tarfile.open(name=filename) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                tar.extract(member=member, path=target)
        tar.close()
    else:
        print(f"{filename} was already extracted")


def create_path(path):
    path = pathlib.Path(path)
    if not os.path.exists(path):
        print(f"Created: {path}")
        path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{path} already exists")


def recursive_remove(dir_path):
    dir_path = pathlib.Path(dir_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed: {dir_path}")


def set_up_environment():
    create_path(folder_structure_cfg.data_root)
    create_path(folder_structure_cfg.log_path)
    create_path(folder_structure_cfg.PDBind_data)
    create_path(folder_structure_cfg.PDBind_download)
    # Paths for extracted files
    # create_path(folder_structure_cfg.pp_root)
    # create_path(folder_structure_cfg.pl_refined_root)
    # create_path(folder_structure_cfg.pl_full_root)

    # Download protein-protein data
    filename = os.path.split(pdbind_urls.protein_protein)[1]
    filename = folder_structure_cfg.PDBind_download.joinpath(filename)
    url = pdbind_urls.protein_protein
    download(url, filename)

    # Download refined set
    filename = os.path.split(pdbind_urls.protein_ligand_refined)[1]
    filename = folder_structure_cfg.PDBind_download.joinpath(filename)
    url = pdbind_urls.protein_ligand_refined
    download(url, folder_structure_cfg.PDBind_data.joinpath(filename))
    # Download full set
    filename = os.path.split(pdbind_urls.protein_ligand_full)[1]
    filename = folder_structure_cfg.PDBind_download.joinpath(filename)
    url = pdbind_urls.protein_ligand_full
    download(url, folder_structure_cfg.PDBind_data.joinpath(filename))

    # Extract downloaded tar.gz files
    extract_targz(folder_structure_cfg.pp_targz, folder_structure_cfg.pp_root)
    extract_targz(folder_structure_cfg.pl_refined_targz, folder_structure_cfg.pl_refined_root)
    extract_targz(folder_structure_cfg.pl_full_targz, folder_structure_cfg.pl_full_root)


def set_up_sql_database():
    pdbind_sql.PDBindDataset(folder_structure_cfg.PDBind_data)


def set_up_dataset_caches():
    sql_db = pdbind_sql.PDBindDataset(config.folder_structure_cfg.PDBind_sql)
    samples = sql_db.get_2chain_samples()
    whole_set = DistogramSequenceDataset(
        samples, 512, set_type="whole", feature_type="stacked")
    t = time.localtime()
    creation_date = f"{t.tm_year}{t.tm_mon:>02}{t.tm_mday:>02}_{t.tm_hour:>02}_{t.tm_min:>02}_{t.tm_sec:>02}"
    cache_dir = config.folder_structure_cfg.data_caches.joinpath(f"cache_1_{creation_date}")
    whole_set.cache_it(cache_dir)
    print("Finished")


def clean_up_environment():
    def validate_input(x):
        if x == "Y" or x == "y":
            return True
        elif x == "N" or x == "n":
            return False
        else:
            return None

    dirs_to_remove = [
        folder_structure_cfg.data_root,
        folder_structure_cfg.log_path
    ]
    print("Following directories will be recursively removed:")
    for p in dirs_to_remove:
        if os.path.exists(p):
            print(f"- {p}")
            for f in list_files_in_dir(p):
                print(f"  -{f}")

    x = validate_input(input("Continue? (y,n)"))
    while x is None:
        x = validate_input(input("Continue? (y,n)"))
    if x:
        for p in dirs_to_remove:
            recursive_remove(p)
    else:
        return


def run():
    args = get_arguments()
    if args.clean:
        clean_up_environment()
    else:
        set_up_environment()
        set_up_sql_database()
        set_up_dataset_caches()


if __name__ == '__main__':
    run()
