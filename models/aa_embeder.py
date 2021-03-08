from __future__ import annotations
# General imports
import os
import sys
import time
import data
import dataclasses as dc
import dacite

import dacite
import inspect
import itertools
import pickle
import math
import pandas as pd
from itertools import product
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import hashlib
# Project specific imports

# Imports from internal libraries
import config
import utils
from mol_readers.pdbind import PDBindDataset, PandasMolStructure
from nn_utils.utils import select_device, MetaInt, MetaFloat, BaseMetaParam
import feature_constructors.categorical as catFeatures

# Typing imports
from typing import TYPE_CHECKING

# if TYPE_CHECKING:

# MODEL CONFIGS
NN_DEVICE = select_device(1)


class AA_OneHotEncoder:
    def __init__(self, ntuple_size):
        self.ntuple_size = ntuple_size
        self.encoding_map = self.aa_one_hot(ntuple_size)
        self.vocab_size = self.encoding_map.shape[0]

    @staticmethod
    def aa_one_hot(l=1):
        aas = pd.read_csv(config.folder_structure_cfg.aminoacids_csv)
        aas["ISO3"] = aas["ISO3"].str.upper()
        aa_comb = list(product(*[aas["ISO3"] for _ in range(l)]))
        encodings = pd.DataFrame(np.eye(len(aa_comb), dtype=int))
        encodings.columns = pd.MultiIndex.from_tuples(aa_comb, names=[f"aa_{i}" for i in range(l)])
        df = encodings.T.reset_index()
        df.insert(loc=l, column="numerical", value=df.index)
        return df

    def encode(self, sequence):
        multy_sequence = pd.DataFrame([sequence["residue"].shift(-i) for i in range(self.ntuple_size)]).T
        multy_sequence.columns = [f"aa_{i}" for i in range(self.ntuple_size)]
        encoded_seq = pd.merge(multy_sequence, self.encoding_map, how="left",
                               on=list(self.encoding_map.columns[:self.ntuple_size].values))
        return encoded_seq

    def decode(self, numerical_sequence, mode="df"):
        available_types = ["df", "list", "string"]
        assert mode in available_types, f"Wrong type: {mode}. Possible types are {available_types}"
        decoded_seq = pd.merge(pd.Series(numerical_sequence, name="numerical").dropna(),
                               self.encoding_map, how="left", on="numerical")
        decoded_seq = decoded_seq[[f"aa_{i}" for i in range(self.ntuple_size)]]
        decoded_seq.replace(np.nan, '', regex=True, inplace=True)
        if mode == "df":
            return decoded_seq
        elif mode == "list":
            return decoded_seq.values.tolist()
        elif mode == "string":
            decoded_list = decoded_seq.values.tolist()
            return list(map(lambda x: ",".join(x), decoded_list))


class CBOA_Dataset(Dataset):
    def __init__(self, context_size, ntuple_size,
                 set_type="whole",
                 split={"train": 0.8, "val": 0.1, "test": 0.1},
                 split_variant=0,
                 init_on_GPU=False
                 ):
        types = ["whole", "train", "val", "test"]
        assert set_type in types, f"Wrong type specified: {set_type}. Possible types are: {types}"
        assert math.isclose(sum(split.values()), 1), f"Split schema: {split} should sum up to 1"

        self.context_size = context_size
        self.ntuple_size = ntuple_size
        self.set_type = set_type
        self.split = split
        self.split_variant = split_variant
        self.n_samples = 0
        self.cache_root = config.folder_structure_cfg.data_caches.joinpath("seq_cache")
        self.sequences_file = self.cache_root.joinpath(f"numerical_sequences_{self.ntuple_size}_tuples.txt")
        self.sample_cache = self.cache_root.joinpath(
            f"samples_{self.ntuple_size}_tuple_{self.context_size}_neighbours.pickle")
        if not os.path.exists(self.cache_root):
            self.cache_root.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.sequences_file):
            print(f"Caching sequences to: {self.sequences_file}")
            self.cache_it()
        else:
            print(f"Sequence cache file allready exists at: {self.sequences_file}")
        self.samples = self.generate_samples()
        self.vocab_size = self._vocab_size_()

        self._split_generator_ = self._split_generator_fun_(verbose=True)
        # self.manager = multiprocessing.Manager()
        self.samples = self.samples[self._get_nth_split_(self.split_variant)].values
        if init_on_GPU and NN_DEVICE == "cuda:0":
            self.samples = torch.from_numpy(self.samples).to(NN_DEVICE)

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
                opt_ratio = n_opt / self.__len__()
                opt_test = opt_ratio > p_opt - epsilon and opt_ratio < p_opt + epsilon
                test_results.append(opt_test)
                ratios.append(opt_ratio)

            i += 1
            if all(test_results):
                if verbose:
                    print(
                        f"Sample seed: {i} OK. Ratios are:{ratios}, Counts: {counts}")
                    if self.set_type == "whole":
                        yield np.full(len(self), True)
                    else:
                        yield choice == self.set_type
            else:
                if verbose:
                    print(
                        f"Sample seed: {i} NOT OK. Ratios are:{ratios}, Counts: {counts}")

    def _get_nth_split_(self, n):
        return next(itertools.islice(self._split_generator_, n, None))

    @staticmethod
    def _sequence_to_string_(pdb_path, queue, ntuple_size):
        pdb_struct = PandasMolStructure()
        pdb_struct.get_pandas_structure(pdb_path)
        seq = pdb_struct.get_protein_sequence()
        encoder = AA_OneHotEncoder(ntuple_size)
        encoded_seq = encoder.encode(seq)
        seq_str = ",".join(str(x) for x in list(encoded_seq["numerical"].dropna().astype(int)))
        queue.put(seq_str)
        return None

    def write_seq_to_file(self, q):
        observed_sequences = set()
        n_saved = 0
        with open(self.sequences_file, 'a') as f:
            while 1:
                seq = q.get(True)
                if seq == 'kill':
                    break
                hex_seq = hashlib.sha256(seq.encode()).hexdigest()

                if hex_seq not in observed_sequences:
                    n_saved += 1
                    print(f"\r# saved: {n_saved} |  Queue len: {q.qsize()}", end="")
                    f.write(seq + '\n')
                    f.flush()
                    observed_sequences.add(hex_seq)

    def cache_it(self):
        sql_db = PDBindDataset(config.folder_structure_cfg.PDBind_sql)
        samples = sql_db.get_all_pdbs()

        open(self.sequences_file, 'w').close()
        samples = samples["protein"]

        mp_manager = multiprocessing.Manager()
        writer_queue = mp_manager.Queue()

        writer_process = multiprocessing.Process(target=self.write_seq_to_file, args=(writer_queue,))
        writer_process.start()

        with utils.Timer("Seq gen:"):
            seq_gen_pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), 25))
            seq_gen_pool.map(partial(self._sequence_to_string_,
                                     queue=writer_queue,
                                     ntuple_size=self.ntuple_size), samples)
            seq_gen_pool.close()
            seq_gen_pool.join()
            writer_queue.put("kill")
            writer_process.join()
            writer_process.terminate()

    @staticmethod
    def samples_from_seq(seq, n_neighbours):
        sample_range = n_neighbours * 2 + 1
        samples = []

        for x in range(len(seq)):
            if len(seq[x:x + sample_range]) == sample_range:
                samples.append([seq[x + n_neighbours]] + seq[x:x + n_neighbours] + seq[
                                                                                   x + n_neighbours + 1:x + n_neighbours * 2 + 1])
        return samples

    def generate_samples(self, caching=True) -> pd.Series:
        if os.path.exists(self.sample_cache):
            print(f"Loading samples from cache at: {self.sample_cache}")
            lines = pickle.load(open(self.sample_cache, "rb"))
        else:
            with open(self.sequences_file, "r") as f:
                lines = f.readlines()
            lines = [list(map(int, x.split(","))) for x in lines]

            pool = multiprocessing.Pool(32)
            lines = pool.map(partial(CBOA_Dataset.samples_from_seq, n_neighbours=self.context_size), lines)
            pool.close()
            pool.join()
            # Continue here change lines to be a pandas dataframe. Problem with memory leak
            lines = list(itertools.chain.from_iterable(lines))
            lines = pd.DataFrame(data=lines, columns=["target"] + [f"Ctx_{x}" for x in range(len(lines[0][1:]))])
            lines.drop_duplicates(inplace=True)
            if caching:
                print(f"Caching samples to: {self.sample_cache}")
                pickle.dump(lines, open(self.sample_cache, "wb"))
        return lines

    def _vocab_size_(self):
        aas = pd.read_csv(config.folder_structure_cfg.aminoacids_csv)
        return len(aas) ** self.ntuple_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        target = self.samples[item][0]
        context = self.samples[item][1:]
        # target, context = self.samples.iloc[item]
        # context = torch.tensor(context, dtype=torch.long)
        # target = torch.tensor(target, dtype=torch.long)

        return target, context


@dc.dataclass
class MetaParams(BaseMetaParam):
    model_params: ModelParams
    optimizer_params: OptimizerParams
    dataset_params: DatasetParams


@dc.dataclass
class OptimizerParams(BaseMetaParam):
    learning_rate: float


@dc.dataclass
class DatasetParams(BaseMetaParam):
    context_size: int
    batch_size: int
    ntuple_size: int


@dc.dataclass
class ModelParams(BaseMetaParam):
    model_name: str
    vocab_size: int
    embedding_dim: int
    context_size: int


class NGramLanguageModeler(nn.Module):
    model_name = "Base embedder"

    def __init__(self, vocab_size, embedding_dim, context_size, **kwargs):
        super(NGramLanguageModeler, self).__init__()
        self.metadata: MetaParams = None
        # TODO Change metadata to be a dataclass. Easyer loading of model parameter
        self.generate_metadata(locals(), inspect.signature(self.__init__))
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def generate_metadata(self, local_scope, init_atributes):
        for k in list(local_scope.keys()):
            if k not in init_atributes.parameters.keys():
                del local_scope[k]
        local_scope["model_name"] = self.model_name
        self.metadata = dacite.from_dict(ModelParams, local_scope)
        return self

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    print(f"Selected device: {NN_DEVICE}")
    utils.logger.init()

    dataset_params = DatasetParams(batch_size=64,
                                   ntuple_size=2,
                                   context_size=8)
    optimizer_params = OptimizerParams(learning_rate=0.001)

    N_EPOCHS = 200

    NUM_WORKERS = 0
    TRAIN_TEST_VAL_SPLIT = {"train": 0.8, "test": 0.1, "val": 0.1}

    train_dataset = CBOA_Dataset(context_size=dataset_params.context_size, ntuple_size=dataset_params.ntuple_size,
                                 set_type="train",
                                 split=TRAIN_TEST_VAL_SPLIT, split_variant=0, init_on_GPU=False)
    validation_dataset = CBOA_Dataset(context_size=dataset_params.context_size, ntuple_size=dataset_params.ntuple_size,
                                      set_type="val",
                                      split=TRAIN_TEST_VAL_SPLIT, split_variant=0, init_on_GPU=False)
    test_dataset = CBOA_Dataset(context_size=dataset_params.context_size, ntuple_size=dataset_params.ntuple_size,
                                set_type="test",
                                split=TRAIN_TEST_VAL_SPLIT, split_variant=0, init_on_GPU=False)

    train_loader = DataLoader(train_dataset, batch_size=dataset_params.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=dataset_params.batch_size, shuffle=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=dataset_params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                             drop_last=True,
                             pin_memory=True)

    loss_function = nn.NLLLoss()

    model_params = ModelParams(model_name=NGramLanguageModeler.model_name,
                               vocab_size=train_dataset.vocab_size,
                               embedding_dim=2,
                               context_size=train_dataset.context_size * 2)
    model = NGramLanguageModeler(**dc.asdict(model_params)).to(NN_DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=optimizer_params.learning_rate)

    best_model_val_acc = -1

    all_params = MetaParams(model_params=model_params,
                            dataset_params=dataset_params,
                            optimizer_params=optimizer_params)
    utils.logger.save_experiment_metadata(**dc.asdict(all_params))

    with utils.Timer("Training loop"):
        for epoch in range(N_EPOCHS):
            model.train()
            total_loss = 0
            for b, batch in enumerate(train_loader):
                targets = batch[0].to(NN_DEVICE, non_blocking=True)
                contexts = batch[1].to(NN_DEVICE, non_blocking=True)

                model.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(contexts)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(log_probs, targets)

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
                utils.logger.log_training(epoch, b, loss)

                acc = torch.sum(torch.max(log_probs, 1)[1] == targets) / targets.shape[0]
                utils.logger.log_training(epoch, b, loss, accuracy=acc)
                print(f"\r Batch: {b}, Loss: {loss.item():.5f}  ||  Accuracy: {acc:.5f}", end="")

            print("\nValidating")
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0
            model.eval()
            with torch.no_grad():
                for b, batch in enumerate(val_loader):
                    targets = batch[0].to(NN_DEVICE, non_blocking=True)
                    contexts = batch[1].to(NN_DEVICE, non_blocking=True)
                    log_probs = model(contexts)
                    loss = loss_function(log_probs, targets)

                    total_val_loss += loss

                    correct = torch.sum(torch.max(log_probs, 1)[1] == targets)
                    acc = correct / targets.shape[0]
                    total_val_correct += correct
                    total_val_samples += targets.shape[0]

                    utils.logger.log_validation(epoch, b, loss, accuracy=acc)
                    print(f"\r Validation batch accuracy: {acc:.5f}", end="")
                val_acc = total_val_correct / total_val_samples
                print(f"Total validation accuracy: {val_acc:.5f}")

            if val_acc >= best_model_val_acc:
                print("Better model found")
                utils.logger.save_model_state(model, "aa_embedder")
                best_model_val_acc = val_acc

    print("\nTesting")

    total_test_loss = 0
    total_test_correct = 0
    total_test_samples = 0
    model.eval()
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            targets = batch[0].to(NN_DEVICE, non_blocking=True)
            contexts = batch[1].to(NN_DEVICE, non_blocking=True)
            log_probs = model(contexts)
            loss = loss_function(log_probs, targets)
            utils.logger.log_validation("", b, loss)
            total_test_loss += loss

            correct = torch.sum(torch.max(log_probs, 1)[1] == targets)
            acc = correct / targets.shape[0]
            total_test_correct += correct
            total_test_samples += targets.shape[0]
            print(f"\r Test batch accuracy: {acc:.5f}", end="")

    print(f"Total test accuracy: {total_test_correct / total_test_samples:.5f}")
