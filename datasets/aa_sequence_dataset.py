# General imports
import os
from sqlite3.dbapi2 import SQLITE_SELECT
from numpy.core.numeric import True_

from numpy.lib.arraysetops import unique
# Project specific imports

# Imports from internal libraries
from mol_readers.pdbind import PDBindDataset, PandasMolStructure
import feature_constructors.categorical as catFeatures
import config
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
    

sql_db = PDBindDataset(config.PDBIND_SQLITE_DB)
samples = sql_db.get_all_pdbs()

pdb_struct = PandasMolStructure()
protein_structure = pdb_struct.get_pandas_structure(samples["protein"][0])

seq = pdb_struct.get_protein_sequence()

class ProteinSequenceDataset:
    def __init__(self,pdb_paths):
        super().__init__()
        self.pdb_paths = pdb_paths
        self.special_tokens = {"sequence_start":"START","sequence_end":"END"}
        self.token_set = set()
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

        








if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    seq_dataset = ProteinSequenceDataset(samples.protein)

    seq_dataset.generate_dataset()
    a = 1