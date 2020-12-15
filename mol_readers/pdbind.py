# Future imports
from __future__ import annotations
# General imports
import os
from os import stat
import sys
import re
from typing import List
import pandas as pd
import numpy as np
import sqlite3
from multiprocessing import Pool

# Project specific imports


# Imports from internal libraries
import config
import utils
from mol_readers.pdb_transforms import PandasMolStructure


class PDBindSample:
    def __init__(self, root):
        self.files = []
        self.root = root

    def parse_name(self):
        _, name = os.path.split(self.root)
        return name

    def absoulute_paths(self):
        return list(map(lambda x: self.root+"/"+x,self.files))

    def __repr__(self):
        return f"PDBind example: {self.parse_name()}"

class PDBindDataset:

    def __init__(self,root):
        self.root = root        
        self.index_files = []
        if self.check_db_exist():
            print("Database allready exists")
            conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
            self.db = pd.read_sql("SELECT * FROM samples",conn)
        else:
            print("Creating new database")
            self.find_index_files(self.find_index_root())
            conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
            self.db = self.parse_index_files()
            self.db.to_sql("samples",conn,if_exists="fail")
    
    def check_db_exist(self)->bool:
        """ Check if SQLite database exists at location, specified
        in the config file.
        Returns:
            bool: True if exists False otherwise
        """
        if os.path.exists(config.PDBIND_SQLITE_DB):
            return True
        else:
            return False
    
    def reset_database(self)->PDBindDataset:
        """Reinitializes the database. Used when some changes are made in database creation procedure.
        Returns:
            PDBindDataset: self
        """
        if self.check_db_exist():
            print("Reseting database")
            self.find_index_files(self.find_index_root())
            conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
            self.db = self.parse_index_files()
            self.db.to_sql("samples",conn,if_exists="replace")
        return self


    def find_index_root(self)->List[str]:
        """Finds the root directory of index files which describe individual samples in the dataset.
        Returns:
            List[str]: list of paths to index roots.
        """

        index_root = []
        idnex_re = re.compile("index",re.IGNORECASE)
        for root,dirs,files in os.walk(self.root):
            for d in dirs:
                if re.search(idnex_re,d):
                    index_root.append(root+"/"+d+"/")
                    print(f"Index root found at: {index_root}")
        if len(index_root)>1:
            print(f"Multiple indices found: {len(index_root)}\n{index_root}")
        return index_root

    def find_index_files(self,index_roots:List[str])->PDBindDataset:
        """
        Finds all idnex files that describe the dataset. Index files that have same content as 
        allready processed files are ignored.
        Args:
            index_roots (List[str]): List of index roots

        Returns:
            PDBindDataset: self with initialized index files
        """
        idx_file_hashes = set()
        for idx_root in index_roots:
            for _,_,files in os.walk(idx_root):
                for f in files:
                    if f[0] != ".":
                        f_path = f"{idx_root}{f}"
                        f_sha = utils.compute_file_sha256(f_path)
                        if f_sha not in idx_file_hashes:
                            idx_file_hashes.add(f_sha)
                            self.index_files.append(f_path)
        return self
    
    def parse_index_files(self)->pd.DataFrame:
        """Parses the index files and extracts binding type and resolution of individual
           examples.
        Returns:
            pd.DataFrame: Data frame with columns: ligand_type,PDB, resolution 
        """

        files_to_parse = {
            "protein-protein":re.compile("INDEX_general_PP.2019",re.IGNORECASE),
            "protein-ligand":re.compile("INDEX_general_PL.2019",re.IGNORECASE)
        }

        # Parse PDBind index files to and construct pandas dataframe from them
        re_multiple_space = re.compile(" +")
        pd_data = []
        for f in self.index_files:
            for ligand_type in files_to_parse:
                file_re = files_to_parse[ligand_type]
                if re.search(file_re,f):
                    with open(f,'r',errors='ignore') as fp:
                        line = fp.readline()
                        cnt = 1
                        while line:
                            if line[0] != "#":
                                data = line.split("//", 1)[0]
                                data = re.sub(re_multiple_space," ",data)
                                data = data.split(" ")[:2]
                                data.insert(0,ligand_type)
                                pd_data.append(data)
                                line = fp.readline()
                                cnt += 1
                            else:
                                line = fp.readline()
        
        pd_data = pd.DataFrame(pd_data,columns=["ligand_type","PDB","resolution"])
        pd_data["num_chains"] = np.NaN
        pd_data["protein"] = np.NaN
        pd_data["ligand"] = np.NaN

        re_protein_protein = re.compile("([a-z0-9]*?)\.ent\.pdb",re.IGNORECASE)
        re_ligand = re.compile("([a-z0-9]*?)_ligand.mol2",re.IGNORECASE)
        re_protein_lingand = re.compile("([a-z0-9]*?)_protein\.pdb",re.IGNORECASE)
        
        mol_types ={
            "protein-protein":re_protein_protein,
            "protein-ligand":re_protein_lingand,
            "ligand":re_ligand
        }

        file_count = 0
        for root,dirs,files in os.walk(self.root):
            for f in files:
                file_count += 1
                print(f"\rProcessing file: {file_count}", end="")
                for mol_type,type_reg in mol_types.items():
                    res = re.search(type_reg,f)
                    # FIXME this part for inserting file path data to the database is inefficeint. Think how to speed
                    # it up. Maybe something in the flawor of creating another data frame and then do a merge
                    if res:
                        pdb_id = res.group(1)
                        if mol_type == "protein-protein" or mol_type == "protein-ligand":
                            pd_data.loc[pd_data['PDB'] == pdb_id,["protein"]] = f"{root}/{f}"                            

                            pdb_transforms = PandasMolStructure()
                            pd_struct = pdb_transforms.get_pandas_structure(f"{root}/{f}")
                            pd_struct = pd_struct[~pd_struct["is_hetatom"]]
                            num_chains = pd_struct.chain.unique().size
                            pd_data.loc[pd_data['PDB'] == pdb_id,["num_chains"]] = num_chains
                        if mol_type == "ligand":
                             pd_data.loc[pd_data['PDB'] == pdb_id,["ligand"]] = f"{root}/{f}"   


        return pd_data            


    # TODO Create different sampling procedures. Random, static (for static store sample sets in the database)
    # TODO Define a list of features. Start with a matrix representation of energies between residues of both molecules
    # TODO Create methods that will return numpy arrays of desired features. These arrays will be consumed by
    # Pytorch abstraction of DataLoaders
    # TODO Feature visualization (Dash app for starters, later it can be converted to Angular+WebGL if Dash tourns out to be clunky)

    def get_samples(self, limit=None)->List[PDBindSample]:
        all_files = []
        curr_processed = 0
        for root, _, files in os.walk(self.root):
            d = PDBindSample(root)
            for f in files:
                if curr_processed <= limit:
                    curr_processed += 1
                    d.files.append(f)
                else:
                    return all_files
            if len(d.files) > 0:    
                all_files.append(d)
        return all_files
    

    def get_PP_samples(self):
        querry = """
        SELECT * FROM samples
        WHERE
            ligand_type = 'protein-protein' AND
            protein IS NOT NULL
        """
        conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
        pp_samples = pd.read_sql(querry,conn)
        
        for index, row in pp_samples.iterrows():
            yield index,row["protein"]
    

    def get_PL_samples(self):
        
        querry = """
        SELECT * FROM samples
            WHERE
                ligand_type = 'protein-ligand' AND
                protein IS NOT NULL AND 
                ligand IS NOT NULL
        """
        conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
        pp_samples = pd.read_sql(querry,conn)
        
        for index, row in pp_samples.iterrows():
            yield index,row["protein"],row["ligand"]
    
    def get_all_pdbs(self):
        querry = """
        SELECT protein FROM samples
        WHERE protein IS NOT NULL;
        """
        conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
        samples = pd.read_sql(querry,conn)
        return samples


    @staticmethod
    def filter_protein_dna(row):
        pdb_transforms = PandasMolStructure()
        pd_struct = pdb_transforms.get_pandas_structure(row)
        pd_struct = pd_struct[~pd_struct["is_hetatom"]]
        num_chains = pd_struct.chain.unique().size
        if num_chains < 2:
            return True
        else:
            return False



    @staticmethod
    def _filter_protein_dna_(row):
        i = row[0]
        row = row[1]
        print(f"\rFiltering file: {i}", end="")
        pdb_transforms = PandasMolStructure()
        pd_struct = pdb_transforms.get_pandas_structure(row["protein"])
        pd_struct = pd_struct[~pd_struct["is_hetatom"]]
        num_chains = pd_struct.chain.unique().size
        if num_chains < 2:
            # mask.append(True)
            return True
        else:
            return False
            # mask.append(False)

    def get_2chain_samples(self)->pd.DataFrame:
        """
        Generator for getting samples that protein-protein samples that have only 2 chains
        Returns:
            pd.DataFrame: [description]
        """        

        q = """
        SELECT * FROM samples 
        WHERE
        num_chains = 2 AND
        ligand_type = "protein-protein"
        """
        conn = sqlite3.connect(config.PDBIND_SQLITE_DB)
        two_chain_samples = pd.read_sql(q,conn)
        pool = Pool(24)
        mask = pool.map(self._filter_protein_dna_, two_chain_samples.iterrows())
        return two_chain_samples[~pd.Series(mask)]

        # Uncomment for generator behaviour
        # for index, row in two_chain_samples.iterrows():
        #     yield index,row["protein"],row["ligand"]

    
    # Dataset augmentations

    # def gen_col_num_chains(self):
    
if __name__ == '__main__':
    print(f'Running {__file__}')

    pdbind_dataset = PDBindDataset(config.PDBIND_DATA_ROOT)

    # pdbind_dataset.reset_database()
    print("Finished")
    PP_samples = pdbind_dataset.get_PP_samples()
    PL_samples = pdbind_dataset.get_PL_samples()

    # with utils.Timer("Yield timer PL samples"):
    #     for i in range(50):
    #         print(next(PP_samples))


    # with utils.Timer("Yield timer PL samples"):
    #     for i in range(50):
    #         print(next(PL_samples))
    
    print(next(PP_samples))

    
