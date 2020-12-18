# General imports
import hashlib
import time
import os
from pathlib import Path
import re
from numpy.core.fromnumeric import clip
import pandas as pd
from typing import List
from pathlib import Path
# Project specific imports
import torch
import cv2
import moviepy.editor as mp
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
    
    def get_experiments(self):
        ignore = {".DS_Store", "._.DS_Store"}
        logs = [f"{self.log_path}/{x}" for x in os.listdir(self.log_path) if x not in ignore]
        return logs

    def gen_filter_act_video(self):
        print("Generating filter activation video")
    
    def gen_weight_distrbi_video(self):
        print("Generating weight distribution video")        
    
    def generate_all_videos(self,log:str):
        print(f"Generating all videeos for log: {log}")
        print(self.get_experiments())
    
    
    def _get_numerical_input_(selfs):
        seletion = input()
        while not seletion.isdigit():
            seletion = input("You must type in an integer number")
        seletion = int(seletion)
        return seletion

    
    def generrate_video_from_path(self,path:str,epoch_identifier:str):
        print(f"Generating video for image: {path}, epoch identifier: {epoch_identifier}")
        left = path.split(epoch_identifier)[0]
        right = path.split(epoch_identifier)[1]
        epochs = [int(x) for x in os.listdir(left)]
        epochs.sort()

        img_array = []
        dim = None
        scale = 0.25
        for e in epochs:
            p = os.path.join(left,str(e),right)            
            img = cv2.imread(p)
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            dim = (width, height)
            img_array.append(cv2.resize(img, dim, interpolation=cv2.INTER_AREA))
        
        out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'H264'), 8, dim)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        # clip = mp.VideoFileClip("project.mp4")
        # clip.write_videofile("project.webm")

                    




    def manual_video_generation(self):
        experiments = self.get_experiments()
        print("Choose experiment:")
        for i,experiment in enumerate(experiments):
            print(f"{i}: {experiment}")
        
        print("Seslect experiment by typing number:")
        # Get numerical input
        exp_sel = self._get_numerical_input_()        
        experiment = experiments[exp_sel]
        print(f"Selected experiment: {exp_sel}: {experiment}")

        print("Select video to generate:")
        options = ["ativations","distributions"]
        for i,opt in enumerate(options):
            print(f"{i}: {opt}")
        opt_sel = self._get_numerical_input_()
        option = options[opt_sel]
        print(f"Selected option: {option}")



        print("Select example:")
        samples = os.listdir(f"{experiment}/nn_vis/{0}")
        for i,sample in enumerate(samples):
            print(f"{i}: {sample}")
        sample_sel = self._get_numerical_input_()
        sample = samples[sample_sel]
        print(f"Selected sample: {sample}")

        print("Select Layer:")
        layers = os.listdir(f"{experiment}/nn_vis/{0}/{sample}")
        for i, layer in enumerate(layers):
            print(f"{i}: {layer}")
        
        layer_sel = self._get_numerical_input_()
        layer = layers[layer_sel]
        print(f"Selected layer: {layer}")

        print("Select filter:")
        filters = os.listdir(f"{experiment}/nn_vis/{0}/{sample}/{layer}")
        for i, filt in enumerate(filters):
            print(f"{i}: {filt}")
        
        filt_sel = self._get_numerical_input_()
        filter = filters[filt_sel]
        print(f"Selected filter: {filter}")

        






        

        




    def __repr__(self):
        s = f"Logger object | Root: {self.current_path}"
        return s

logger = TrainLogger(config.LOG_PATH)

if __name__ == "__main__":
    path = "/home/erikj/projects/insidrug/py_proj/erikj/loggs/2020_12_01_12_06_38_latest/nn_vis/0/25_1buh_ent_pdb/downscale.cnn_1/filt_0.png"
    logger.generrate_video_from_path(path,"/0/")
    


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
