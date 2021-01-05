# General imports
import os
import numpy as np
import pandas as pd
import re
# Project specific imports
from Bio.PDB import PDBParser, Structure
from sklearn.metrics.pairwise import euclidean_distances
# Imports from internal libraries
import utils
import config_old
import config


class PandasMolStructure:
    def __init__(self):
        self.parser = PDBParser(QUIET=True, PERMISSIVE=True)
        self.df_structure = None
        self.pairwise_dist = None

    def get_pandas_structure(self, pdb_file: str = None,het_atom =False) -> pd.DataFrame:
        # TODO split get and init method
        """Constructs a pandas.DataFrame representation of PDB protein structure
        Args:
            pdb_file (str): Path to PDB file

        Returns:
            pd.DataFrame: DataFrame with following strudcture
            {"model":[],"chain":[],"residue":[],"atom":[],"x":[],"y":[],"z":[]}
        """
        assert not (self.df_structure is None and pdb_file is None),\
            "Data has not been initialized yet and no pdb file was provided"

        if self.df_structure is None:
            df_dict = {"model": [], "chain": [], "residue": [], "res_pos": [],
                       "atom": [],"atom_pos":[],"is_hetatom":[], "x": [], "y": [], "z": []}

            structure = self.parser.get_structure("protein_1", pdb_file)

            # TODO Option should be given to cohoose if we want to use all models from NMR pdb_samples
            model = next(structure.get_models())
            # for model in structure.get_models():
            for chain in model.get_chains():
                for residue in chain.get_residues():                    

                    for atom in residue.get_atoms():
                        df_dict["model"].append(model.id)
                        df_dict["chain"].append(chain.id)
                        df_dict["residue"].append(residue.get_resname())
                        df_dict["res_pos"].append(residue.id[1])
                        df_dict["atom"].append(atom.get_name())
                        df_dict["atom_pos"].append(atom.serial_number)
                        df_dict["is_hetatom"].append(not bool(re.search('het= ',residue.__repr__())))
                        cords = atom.get_coord()
                        df_dict["x"].append(cords[0])
                        df_dict["y"].append(cords[1])
                        df_dict["z"].append(cords[2])
            self.df_structure = pd.DataFrame(df_dict)
        
        if het_atom:
            return self.df_structure
        else:
            # het_atoms_to_ignore = ["HOH","NAG", "FUC", "MAN", "GAL", "SO4"]
            # FIXME Atoms to ignore should be based on HETATOM
            atoms_to_not_ignore = utils.get_AA_list(config.folder_structure_cfg.aminoacids_csv)
            return self.df_structure[self.df_structure["residue"].isin(atoms_to_not_ignore)]

    def get_atom_3Dcoord(self, pdb_file: str) -> np.array:
        """Returns numpy array of 3D atom positions
        Args:
            pdb_file (str): Path to pdb file

        Returns:
            np.array: Array of shape Nx3 where N is the number of atoms
            in pdb_file
        """
        # TODO filter out heteroatoms (watter)
        structure = self.parser.get_structure("protein_1", pdb_file)
        atoms = []
        for atom in structure.get_atoms():
            cords = atom.get_coord()
            atoms.append(cords)
        return np.array(atoms)

    
    def get_protein_sequence(self):
        structure_df = self.get_pandas_structure()
        sequence = structure_df[structure_df['residue'].shift() != structure_df['residue']]\
            .reset_index(drop=True)[["residue", "res_pos"]]
        return sequence
        
    

    @staticmethod
    def get_pairwise_euclidean_atom(sturcture_df:pd.DataFrame):#pdb_file: str = None,het_atom=False):
        # TODO upgrade distances to energy calculations based on distance and charge
        # TODO find data for atom charges
        # TODO whta to do with missing hydrogen atoms? X-Ray doesent determine H positions

        # if self.pairwise_dist is None:
        cords = sturcture_df[["x", "y", "z"]]        
        return euclidean_distances(cords, cords)


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir:{os.path.abspath(os.getcwd())}")

    test_pdb_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1acb.ent.pdb"

    pdb_transforms = PandasMolStructure()
    pdb_transforms.get_pandas_structure(test_pdb_file)
    a = pdb_transforms.get_atom_3Dcoord(test_pdb_file)
