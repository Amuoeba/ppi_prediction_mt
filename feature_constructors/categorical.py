# General imports
import os
import numpy as np
import pandas as pd
from typing import List
# Project specific imports
import config

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mol_readers.pdb_transforms import PandasMolStructure


def consecutive(data: np.ndarray, stepsize=0) -> List[np.ndarray]:
    """Generate list of arrays where each array contains consecutive elements that are at most 'stepsize' apart.
    Args:
        data (np.array): input array. 1D array of a sequence 
        stepsize (int, optional): How far apart can elements be to still consider them sequential. Defaults to 0.
    Returns:
        [type]: [description]
    """
    b = np.roll(data,1)    
    return np.split(data,np.where((data!=b)[1:])[0]+1)


def resize_categorical(in_cat: np.ndarray, target: int) -> np.ndarray:
    """Resizes a vector of 1D categorical features to be of lenght 'target'. Usefull when categorical data
    comes in in chunks where values inside chunks are of same category (some sequential atoms belong to same
    amino acid when reading a PDB file) adn you need to resize the data to resemble the input image shape.
    Args:
        in_cat (np.ndarray): categorical vector[description]
        target (int): target size of a vector[description]

    Returns:
        np.ndarray: [description]

    EXAMPLE:
    in_cat => [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,3,3,3,3,3,4,4,4,1,1,1,1,1,5,5,5,6,7,7,7,7,7]
    target => 20
    out    => [1,1,2,2,2,2,2,2,2,2,1,1,3,3,4,1,1,5,6,7,7]
    """
    ratio = target/len(in_cat)
    consec = consecutive(in_cat)
    elements = np.array(list(map(lambda x: x[0], consec)))
    lenghts = np.array(list(map(lambda x: len(x)*ratio, consec)))
    new_lenghts = np.clip(lenghts, 1, None)
    new_lenghts = np.rint(new_lenghts).astype('int64')

    len_sum = sum(new_lenghts)
    while len_sum != target:
        if len_sum < target:
            min_ind = np.argmin(new_lenghts)
            new_lenghts[min_ind] = new_lenghts[min_ind] + 1
            len_sum += 1
        elif len_sum > target:
            max_ind = np.argmax(new_lenghts)
            new_lenghts[max_ind] = new_lenghts[max_ind] - 1
            len_sum -= 1
        else:
            continue
    new_el = np.repeat(elements, new_lenghts)

    return new_el


def AA_to_ordinal(aa_file: str, pdb_mol_struct: "PandasMolStructure"):
    """

    Args:
        aa_file (str): [description]
        pdb_mol_struct (PandasMolStructure): [description]

    Returns:
        [type]: [description]
    """
    aas = pd.read_csv(aa_file)
    aas["ordinal_zero_one"] = np.linspace(0, 1, num=aas.shape[0])
    aas["ISO3"] = aas["ISO3"].str.upper()
    sequence = pd.merge(pdb_mol_struct.get_pandas_structure(), aas,
                        how="left",
                        left_on=["residue"],
                        right_on=["ISO3"])[["residue","chain", "ordinal_zero_one"]]
    return sequence


def get_2D_feature_map(sequence,target_size):
    resized_seq = resize_categorical(sequence,target_size)
    horizontal = np.tile(resized_seq,target_size).reshape((target_size,target_size))

    vertical =np.flip(horizontal.T)
    return horizontal,vertical

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
    from mol_readers.pdb_transforms import PandasMolStructure
    import cv2

    test_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1acb.ent.pdb"
    pdb_struct = PandasMolStructure()
    pdb_struct.get_pandas_structure(test_file)

    seq = AA_to_ordinal(config.AMINO_ACIDS, pdb_struct)
    seq1 = seq[seq["chain"] == seq["chain"].unique()[0]
                   ]["residue"]

    seq1 = seq1.to_numpy()
    features = get_2D_feature_map(seq1,512)
