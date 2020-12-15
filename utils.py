# General imports
import hashlib
import time
import os
from pathlib import Path
import re
import pandas as pd
from typing import List
from pathlib import Path
# Project specific imports
import torch
# Imports from internal libraries
from visualizations.heatmaps import output_target_heatmaps
import config


def compute_file_sha256(file):
    """Computes sha256 hash for content of 'file'

    Args:
        file (str): Path to file

    Returns:
        str: Hexdigest of sha256
    """
    BLOCK_SIZE = 65536
    # Create the hash object, can use something other than `.sha256()` if you wish
    file_hash = hashlib.sha256()
    with open(file, 'rb') as f:  # Open the file to read it's bytes
        # Read from the file. Take in the amount declared above
        fb = f.read(BLOCK_SIZE)
        while len(fb) > 0:  # While there is still data being read from the file
            file_hash.update(fb)  # Update the hash
            fb = f.read(BLOCK_SIZE)  # Read the next block from the file
    return file_hash.hexdigest()


def compare_files(file1, file2) -> bool:
    """
    Checks if 2 files have same content by calculating sha256 and comparing the hash
    Args:
        file1 (string): Paths to file1
        file2 (string): Paths to file2
    Returns:
        bool: True if files have same content, False otherwise
    """
    hash1 = compute_file_sha256(file1)
    hash2 = compute_file_sha256(file2)

    if hash1 == hash2:
        return True
    else:
        return False


def chunk_large_PDB(pdb_file, dest_dir):
    # TODO Add abbility to specyfy how large you want the smaller files to be
    """Chunks a large .pdb file into smaller ones so that PyMol can open them with no problem.
    In current implementation it chunks 100 timesteps together, but this should be changed to
    have more configurability.

    Args:
        pdb_file (str): path to large .pdb file
        dest_dir (str): directory where smaller .pdb's will be created. 
        If it doesen't exist the directory will be created
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    re_endmdl = re.compile("ENDMDL")
    with open(pdb_file) as fp:
        line = fp.readline()
        file_count = 0
        model_count_aux = 0
        small_name = f"{dest_dir}small_out_{file_count}.pdb"

        small_fp = open(small_name, "a")

        while line:
            line = fp.readline()
            if len(line) > 0:
                if line[0] == "E":
                    if re.search(re_endmdl, line) and model_count_aux >= 100:
                        small_fp.write(line)
                        small_fp.close()
                        model_count_aux = 0
                        file_count += 1
                        small_name = f"{dest_dir}small_out_{file_count}.pdb"
                        small_fp = open(small_name, "a")
                    else:
                        small_fp.write(line)
                        model_count_aux += 1
                else:
                    small_fp.write(line)


class Timer(object):
    """Timer context to check how long a block of code takes to complete. Use like this:
    with Timer('Your custom name'):
        arbitrary code block
    Args:
        object ([type]): [description]
    """

    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print(f"{self.description}: {self.end - self.start}")


def get_AA_list(aa_file: str) -> List[str]:
    df = pd.read_csv(aa_file)
    return list(df["ISO3"].str.upper())


# NN utils
def get_num_params(torch_module):
    return sum(p.numel() for p in torch_module.parameters() if p.requires_grad)


class TrainLogger:
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path        
        t = time.localtime()
        self.current_log = f"{t.tm_year}_{t.tm_mon:>02}_{t.tm_mday:>02}_{t.tm_hour:>02}_{t.tm_min:>02}_{t.tm_sec:>02}_latest"
        self.current_path = f"{self.log_path}/{self.current_log}"
        self.nn_vis_path = f"{self.current_path}/nn_vis"
        self.image_path = f"{self.current_path}/images"
        self.train_log = f"{self.current_path}/train_log.txt"
        self.val_log = f"{self.current_path}/val_log.txt"
        self.test_log = f"{self.current_path}/test_log.txt"

        # Create paths
        # Path(self.current_path).mkdir(parents=True, exist_ok=True)
        # Path(self.image_path).mkdir(parents=True, exist_ok=True)

        # Init logg files
        # self._init_files_()

    def init(self):
        self.remove_latest_tag()
        Path(self.current_path).mkdir(parents=True, exist_ok=True)
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        self._init_files_()
    
    def _init_files_(self):
        with open(self.train_log,"w+") as tl:
            cols = "timestamp,epoch,batch,loss,learning_rate"
            tl.write(cols+os.linesep)
        with open(self.val_log,"w+") as vl:
            cols = "timestamp,epoch,batch,loss,learning_rate"
            vl.write(cols+os.linesep)
        
    
    def log_training(self,epoch,batch,loss,lr = ""):
        t = time.localtime()
        timestamp = f"{t.tm_year}_{t.tm_mon:>02}_{t.tm_mday:>02}_{t.tm_hour:>02}_{t.tm_min:>02}_{t.tm_sec:>02}"
        with open(self.train_log, "a+") as myfile:
            row = f"{timestamp},{epoch},{batch},{loss},{lr}"
            myfile.write(row+os.linesep)
    
    def log_validation(self,epoch,batch,loss):
        t = time.localtime()
        timestamp = f"{t.tm_year}_{t.tm_mon:>02}_{t.tm_mday:>02}_{t.tm_hour:>02}_{t.tm_min:>02}_{t.tm_sec:>02}"
        with open(self.val_log, "a+") as myfile:
            row = f"{timestamp},{epoch},{batch},{loss}"
            myfile.write(row+os.linesep)

    def save_target_VS_output(self, target, output, filename, epoch, batch):
        image_name = f"{self.image_path}/{epoch}_{batch}_{filename}.png"
        img = output_target_heatmaps(target, output)
        print(f"Saving image: {image_name}")
        img.write_image(image_name)
    
    def remove_latest_tag(self):
        for d in os.listdir(self.log_path):
            if "_latest" in d:
                old_dir = f"{self.log_path}/{d}"
                new_dir = f"{self.log_path}/{d.replace('_latest','')}"
                os.rename(old_dir,new_dir)

    def __repr__(self):
        s = f"Logger object | Root: {self.current_path}"
        return s

logger = TrainLogger(config.LOG_PATH)

if __name__ == "__main__":
    logger

    # a = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileA.txt"
    # b = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileB.txt"
    # c = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileC.txt"
    # t = "/home/erikj/projects/PDBind_exploration/data/v2019-other-PL/index/2019_index.lst"
    # compute_file_sha256(a)
    # compute_file_sha256(b)
    # compute_file_sha256(c)
    # print(compare_files(a,a))
    # print(compare_files(a,b))
    # print(compare_files(a,c))

    # chunk_large_PDB("/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/argon_output.pdb","/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/smal_pdbs/")
