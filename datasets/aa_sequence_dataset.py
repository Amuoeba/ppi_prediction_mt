# General imports
import os
from sqlite3.dbapi2 import SQLITE_SELECT
from numpy.core.numeric import True_
import multiprocessing
from numpy.lib.arraysetops import unique
# Project specific imports

# Imports from internal libraries
from mol_readers.pdbind import PDBindDataset, PandasMolStructure
import feature_constructors.categorical as catFeatures
import config_old
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
    

sql_db = PDBindDataset(config.folder_structure_cfg.PDBind_sql)
samples = sql_db.get_all_pdbs()

pdb_struct = PandasMolStructure()
protein_structure = pdb_struct.get_pandas_structure(samples["protein"][0])

seq = pdb_struct.get_protein_sequence()

class ProteinSequenceDataset:
    def __init__(self,pdb_paths):
        super().__init__()
        self.pdb_paths = pdb_paths
        self.special_tokens = {"sequence_start":"START","sequence_end":"END"}
        self.token_set = set(list(self.special_tokens.values()))
        self.sequences = []

    
    def get_protein_sequence(self,chain):
        structure_df = chain
        sequence = structure_df[structure_df['residue'].shift() != structure_df['residue']]\
            .reset_index(drop=True)[["residue", "res_pos"]]
        return sequence

    def get_chain_sequences(self,pdb:str,unique = True):
        """Gets sequences for each individual chain in the PDB file
        Args:
            pdb (str): [description]
            unique (bool, optional): If true returns only unique sequences. Defaults to True.

        Returns:
            chains (List): List of chains
        """

        pdb_struct = PandasMolStructure()
        protein_structure = pdb_struct.get_pandas_structure(pdb)
        protein_structure.groupby("chain")
        chains = [self.get_protein_sequence(x) for _,x in protein_structure.groupby("chain")]
        if unique:
            # Keep only chains that are different
            unique_chains = []
            for c in chains:
                if len(unique_chains) == 0:
                    unique_chains.append(c)
                else:
                    for uc in unique_chains:
                        if not c.equals(uc):
                            unique_chains.append(c)
            chains = unique_chains
        return chains

    def generate_dataset(self):
        for i,pdb in enumerate(self.pdb_paths):
            print(f"Extracting sequence data from {i}: {pdb}\r",end="")
            unique_chains = self.get_chain_sequences(pdb,unique=True)
            for uc in unique_chains:
                unique_residues = set(uc.residue.unique())
                self.token_set = self.token_set.union(unique_residues)
                uc_seq = list(uc.residue)
                uc_seq.insert(0,self.special_tokens["sequence_start"])
                uc_seq.append(self.special_tokens["sequence_end"])
                self.sequences.append(uc_seq)
        return self

    
    def _generate_dataset_mp_aux_(self,pdb):
        i,pdb = pdb
        print(f"Extracting sequence data from {i}: {pdb}\r",end="")
        unique_chains = self.get_chain_sequences(pdb,unique=True)
        chains = []
        for uc in unique_chains:            
            unique_residues = set(uc.residue.unique())
            uc_seq = list(uc.residue)
            uc_seq.insert(0,self.special_tokens["sequence_start"])
            uc_seq.append(self.special_tokens["sequence_end"])

            uc_data = {"seq":uc_seq,"unique_residues":unique_residues}
            chains.append(uc_data)        
        return chains

    def generate_dataset_mp(self):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        results = pool.map(self._generate_dataset_mp_aux_,list(enumerate(self.pdb_paths)))
        for r in results:
            for chain in r:
                self.sequences.append(chain["seq"])
                self.token_set = self.token_set.union(chain["unique_residues"])
        return self

        








if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    seq_dataset = ProteinSequenceDataset(samples.protein)

    # seq_dataset.generate_dataset()
    seq_dataset.generate_dataset_mp()
    a = 1