# General imports
import os
import math
import pandas as pd
import numpy as np
import itertools
import time
import glob
# Project specific imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# Imports from internal libraries
from mol_readers.pdbind import PDBindDataset, PandasMolStructure
import config
from feature_constructors import categorical, distogram_features
import visualizations.heatmaps as vis_hm

# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DistogramSequenceDataset(Dataset):
    """Dataset used for first iteration of the model where we try to predict interaction distogram
    based on 2 protein distograms and sequence matrices of these 2 proteins. 

    Args:
        Dataset ([type]): [description]
    """
    dataset_name = "DistogramSequenceDataset"

    def __init__(self, data_table: pd.DataFrame,
                 shape: int,
                 set_type="whole",
                 feature_type="split",
                 split={"train": 0.8, "val": 0.1, "test": 0.1},
                 split_variant=0,
                 ):

        types = ["whole", "train", "val", "test"]
        feature_types = ["split", "stacked"]
        assert set_type in types, f"Wrong type specified: {set_type}. Possible types are: {types}"
        assert feature_type in feature_types, f"Wrong feature type: {feature_type}. Possible are {feature_types}"
        assert math.isclose(sum(split.values()),1), f"Split schema: {split} should sum up to 1"
        self.set_type = set_type
        self.feature_type = feature_type
        self.split = split
        self.data_table = data_table
        self.shape = shape
        self.split_variant = split_variant
        self._split_generator_ = self._split_generator_fun_(verbose=True)
        if self.set_type != "whole":
            self.data_table = self.data_table[self._get_nth_split_(
                self.split_variant)]

    def _split_generator_fun_(self, verbose=False):
        i = 0
        while True:
            np.random.seed(i)
            choice = np.random.choice(
                list(self.split.keys()), self.__len__(), p=list(self.split.values()))
            (unique, counts) = np.unique(choice, return_counts=True)
            counts = dict(zip(unique, counts))
            epsilon = 0.02

            test_results = []
            ratios = []
            gen_split = (x for x in self.split if self.split[x] > 0.0)
            for t in gen_split:
                try:
                    p_opt = self.split[t]
                    n_opt = counts[t]
                except:
                    test_results.append(False)
                    ratios.append("Value missing")
                opt_ratio = n_opt/self.__len__()
                opt_test = opt_ratio > p_opt-epsilon and opt_ratio < p_opt + epsilon
                test_results.append(opt_test)
                ratios.append(opt_ratio)

            i += 1
            if all(test_results):
                if verbose:
                    print(
                        f"Sample seed: {i} OK. Ratios are:{ratios}, Counts: {counts}")
                yield choice == self.set_type
            else:
                if verbose:
                    print(
                        f"Sample seed: {i} NOT OK. Ratios are:{ratios}, Counts: {counts}")

    def _get_nth_split_(self, n):
        return next(itertools.islice(self._split_generator_, n, None))

    @staticmethod
    def visualize_features(sample, image_dir, verbose=True):
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        print(f"Sample name: {sample['pdb_path']}")
        for k in sample:
            if k != "pdb_path":
                name = f"{k}.png"
                fig = vis_hm.protein_distogram_heatmap(sample[k])
                if verbose:
                    print(f"Saving image: {image_dir}/{name}")
                fig.write_image(f"{image_dir}/{name}")
        return True

    @staticmethod
    def visualize_stacked_features(sample, image_dir, verbose=True):
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        feature = sample["feature"]
        for i, f in enumerate(feature):
            name = f"{i}.png"
            fig = vis_hm.protein_distogram_heatmap(f)
            if verbose:
                print(f"Saving image: {image_dir}/{name}")
            fig.write_image(f"{image_dir}/{name}")

    @staticmethod
    def visualize_batch(batch, image_dir, verbose=True):
        batch_size = len(batch["pdb_path"])
        for i in range(batch_size):
            for k in batch:
                if k != "pdb_path":
                    name = f"{k}.png"
                    fig = vis_hm.protein_distogram_heatmap(batch[k][i])
                    dirname = f"{image_dir}/batch_{i}"
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    if verbose:
                        print(f"Saving image: {dirname}/{name}")
                    fig.write_image(f"{dirname}/{name}")

    def cache_it(self, cache_dir):

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        for i in range(len(self)):
            sample = self[i]
            print(f"sample:{i}, {sample['pdb_path']}")
            _, filename = os.path.split(sample["pdb_path"])
            sample_dir_name = f"{i}_{filename}"
            sample_path = f"{cache_dir}/{sample_dir_name}"
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            feature = sample["feature"].numpy()
            target = sample["dg_inter"].numpy()
            np.save(f"{sample_path}/feature.npy", feature)
            np.save(f"{sample_path}/target.npy", target)

    def __len__(self):
        return self.data_table.shape[0]

    def __getitem__(self, idx):
        row = self.data_table.iloc[idx]

        file = row["protein"]
        # print(file)
        pdb_structure = PandasMolStructure()
        pdb_structure.get_pandas_structure(file)
        # print(pdb_structure.get_pandas_structure()["model"].unique())

        # Categorical features
        seq = categorical.AA_to_ordinal(config.folder_structure_cfg.aminoacids_csv, pdb_structure)
        seq1 = seq[seq["chain"] == seq["chain"].unique()[0]
                   ]["ordinal_zero_one"]
        seq2 = seq[seq["chain"] == seq["chain"].unique()[1]
                   ]["ordinal_zero_one"]
        seq1 = seq1.to_numpy()
        seq2 = seq2.to_numpy()
        cat_horizontal_1, cat_vertical_1 = categorical.get_2D_feature_map(
            seq1, self.shape)
        cat_horizontal_2, cat_vertical_2 = categorical.get_2D_feature_map(
            seq2, self.shape)

        # Distogram features
        dg_ch1, dg_ch2, dg_inter = distogram_features.distogram_2cahin(
            pdb_structure, self.shape)

        if self.feature_type == "split":
            sample = {
                "pdb_path": file,
                "dg_ch1": dg_ch1,
                "dg_ch2": dg_ch2,
                "dg_inter": dg_inter,
                "cat_horizontal_1": cat_horizontal_1,
                "cat_vertical_1": cat_vertical_1,
                "cat_horizontal_2": cat_horizontal_2,
                "cat_vertical_2": cat_vertical_2
            }
        elif self.feature_type == "stacked":

            feature = np.array([
                dg_ch1, cat_horizontal_1, cat_vertical_1,
                dg_ch2, cat_horizontal_2, cat_vertical_2
            ])

            feature_trans = transforms.Compose([
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Normalize(mean=feature.mean(axis=(1, 2)),
                                     std=feature.std(axis=(1, 2)))
            ])

            target_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=dg_inter.mean(),
                                     std=dg_inter.std())
            ])

            sample = {
                "pdb_path": file,
                "feature": feature_trans(feature),
                "dg_inter": target_trans(dg_inter)
            }
        else:
            raise ValueError("Unknown feature_type")

        return sample

    def __str__(self):
        big_center = 15
        small_center = 10
        s = f"Dataset object: {type(self).__name__} \n"
        s += f"{'Lenght:':^{small_center}}|{'Feature size:':^{big_center}}|{'Set type:':^{big_center}}|{'Split type:':^{big_center}}|{'train:':^{small_center}}|{'val:':^{small_center}}|{'test:':^{small_center}}\n"
        s += f"{'-'*(3*big_center + 4*small_center)}\n"
        s += f"{self.__len__():^{small_center}}|{self.shape:^{big_center}}|{self.set_type:^{big_center}}|{self.feature_type:^{big_center}}|{self.split['train']:^{small_center}}|{self.split['val']:^{small_center}}|{self.split['test']:^{small_center}}\n"
        return s


class CachedDistogramSequenceDataset(Dataset):
    def __init__(self, cache_dir,set_type="whole",
                 split={"train": 0.8, "val": 0.1, "test": 0.1},
                 split_variant=0) -> None:

        types = ["whole", "train", "val", "test"]
        assert set_type in types, f"Wrong type specified: {set_type}. Possible types are: {types}"
        assert math.isclose(sum(split.values()),1), f"Split schema: {split} should sum up to 1"
        
        self.cache_dir = cache_dir
        self.samples = pd.Series(os.listdir(self.cache_dir))
        self.set_type = set_type
        self.split=split
        self.split_variant = split_variant
        self._split_generator_ = self._split_generator_fun_(verbose=True)
        self.samples = self.samples[self._get_nth_split_(self.split_variant)]

    def _split_generator_fun_(self, verbose=False):
        i = 0
        while True:
            np.random.seed(i)
            choice = np.random.choice(
                list(self.split.keys()), self.__len__(), p=list(self.split.values()))
            (unique, counts) = np.unique(choice, return_counts=True)
            counts = dict(zip(unique, counts))
            epsilon = 0.02

            test_results = []
            ratios = []
            gen_split = (x for x in self.split if self.split[x] > 0.0)
            for t in gen_split:
                try:
                    p_opt = self.split[t]
                    n_opt = counts[t]
                except:
                    test_results.append(False)
                    ratios.append("Value missing")
                opt_ratio = n_opt/self.__len__()
                opt_test = opt_ratio > p_opt-epsilon and opt_ratio < p_opt + epsilon
                test_results.append(opt_test)
                ratios.append(opt_ratio)

            i += 1
            if all(test_results):
                if verbose:
                    print(
                        f"Sample seed: {i} OK. Ratios are:{ratios}, Counts: {counts}")
                yield choice == self.set_type
            else:
                if verbose:
                    print(
                        f"Sample seed: {i} NOT OK. Ratios are:{ratios}, Counts: {counts}")

    def _get_nth_split_(self, n):
        return next(itertools.islice(self._split_generator_, n, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        curent_sample_dir = f"{self.cache_dir}/{self.samples.iloc[index]}"
        feature = f"{curent_sample_dir}/feature.npy"
        target = f"{curent_sample_dir}/target.npy"
        feature_ar = np.load(feature)
        target_ar = np.load(target)

        sample = {
            "pdb_path": feature,
            "feature": torch.from_numpy(feature_ar),
            "dg_inter": torch.from_numpy(target_ar)
        }
        return sample



if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    sql_db = PDBindDataset(config.folder_structure_cfg.PDBind_sql)
    samples = sql_db.get_2chain_samples()

    train_set = DistogramSequenceDataset(samples, 500, set_type="train")
    # val_set = DistogramSequenceDataset(samples, 500, set_type="val")
    # test_set = DistogramSequenceDataset(samples, 500, set_type="test")

    train_set.visualize_features(
        train_set[25], "/home/erik/Projects/master_thesis/ppi_prediction_mt/test_samples/sample_25")
    # dataset.visualize_features(dataset[1],"/home/erikj/projects/insidrug/py_proj/erikj/notebooks/sample_images/sample_1")
    # dataset.visualize_features(dataset[10],"/home/erikj/projects/insidrug/py_proj/erikj/notebooks/sample_images/sample_10")
    # dataset.visualize_features(dataset[50],"/home/erikj/projects/insidrug/py_proj/erikj/notebooks/sample_images/sample_50")
    # dataset.visualize_features(dataset[224],"/home/erikj/projects/insidrug/py_proj/erikj/notebooks/sample_images/sample_224")

    # stacked_train = DistogramSequenceDataset(samples, 500, set_type="train",
    #                                          feature_type="stacked")
    # stacked_train.visualize_stacked_features(
    #     stacked_train[25], "/home/erikj/projects/insidrug/py_proj/erikj/notebooks/sample_images/sample_25_stacked")

    # for i,sample in enumerate(train_set):
    #     print(i)


    # print("Testing cached dataset")
    # cache = "/home/erikj/projects/insidrug/py_proj/erikj/data/caches/usable_cache_1"
    # cached_train = CachedDistogramSequenceDataset(cache,"train")
    # print("Finished")



    # print("Caching dataset")
    whole_set = DistogramSequenceDataset(
        samples, 512, set_type="whole", feature_type="stacked")
    whole_set.cache_it("./data/caches/chace2")
    print("Finished")
