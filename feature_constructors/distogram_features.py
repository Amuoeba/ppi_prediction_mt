# General imports
import os
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from typing import Tuple
# Project specific imports

# Imports from internal libraries


def visualize_features(features:Tuple[np.ndarray],folder):
    assert len(features) == 3, "Tthis method is ment to visualize 2 cahin features"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i,x in enumerate(features):
        if i<2:
            name = f"chain_{i}"
        else:
            name = "interaction"
        trace = dict(type='heatmap', z=x, colorscale='Picnic')
        layout = dict(width=700, height=700)
        fig1 = go.Figure(data=[trace],layout=layout)
        fig1.write_image(f"{folder}/{name}.jpeg")


def distogram_2cahin(pd_mol_struct:"PandasMolStructure",target_shape:int)->Tuple[np.ndarray]:
    """Genereates a touple of scaled features for 2 chain protein complexes.
    Args:
        pd_mol_struct (PandasMolStructure): 
        target_shape (int): desired shape in pixels. Currently all data is transformed to squares.
    Returns:
        tuple(np.ndarray): tuple[0] = chain1, tuple[1] = chain2, tuple[2] = interaction,
    """    

    struct_df = pd_mol_struct.get_pandas_structure()
    assert struct_df["chain"].unique().shape[0] == 2, \
    """
    PDB file contains more than 2 chains.
    Currenyly only complexes with 2 chains are permited.
    """

    resized_features = []

    # resize chain distograms
    chain1_len = None
    for i,c in enumerate(struct_df.chain.unique()):
        chain_df = struct_df[struct_df["chain"] == c]
        if i == 0:
            chain1_len = chain_df.shape[0]
        c_pw_dist = pd_mol_struct.get_pairwise_euclidean_atom(chain_df)
        resized = cv2.resize(c_pw_dist, (target_shape,target_shape), interpolation = cv2.INTER_LINEAR)
        resized_features.append(resized)

    # resize interaction surface distogram
    pw_dist = pd_mol_struct.get_pairwise_euclidean_atom(struct_df)
    inter_dist = pw_dist[:chain1_len,chain1_len:]
    resized_inter_dist = cv2.resize(inter_dist, (target_shape,target_shape), interpolation = cv2.INTER_LINEAR)
    resized_features.append(resized_inter_dist)


    return tuple(resized_features)



if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
    from mol_readers.pdb_transforms import PandasMolStructure

    test_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1acb.ent.pdb"
    pdb_struct = PandasMolStructure()
    pdb_struct.get_pandas_structure(test_file)
    f = distogram_2cahin(pdb_struct,200)
    visualize_features(f,"./notebooks/feature_test3")
