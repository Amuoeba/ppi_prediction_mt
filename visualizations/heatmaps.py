# General imports
import os
from typing import Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd



# Project specific imports

# Imports from internal libraries


def protein_distogram_heatmap(pairwise_distance: np.ndarray, clip_min=None, clip_max=None) -> go.Figure:
    # metadata = pdb_data
    # test_hmap = structure.get_pairwise_euclidean_atom(metadata)

    if clip_max is not None or clip_min is not None:
        pairwise_distance = np.clip(pairwise_distance, clip_min, None)
    if clip_max is not None or clip_min is not None:
        pairwise_distance[pairwise_distance > clip_max] = 100

    fig1 = go.Figure(data=go.Heatmap(
        z=pairwise_distance,
        x=list(range(pairwise_distance.shape[0])),
        y=list(range(pairwise_distance.shape[1])),
        # customdata = ,
        hoverongaps=False))
    fig1.update_layout(
        autosize=False,
        width=800,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
    return fig1


def output_target_heatmaps(target, output):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Target", "Output"))

    fig.add_trace(
        go.Heatmap(
            z=target,
            x=list(range(target.shape[0])),
            y=list(range(target.shape[1])),
            # customdata = ,
            hoverongaps=False),
        row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=output,
            x=list(range(output.shape[0])),
            y=list(range(output.shape[1])),
            # customdata = ,
            hoverongaps=False),
        row=1, col=2
    )
    fig.update_layout(height=600, width=800,
                      title_text="Side By Side Subplots")
    return fig



def train_val_figure(path):
    df = pd.read_csv(path)
    a = 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["timestamp"],y=df["loss"])
    )
    return fig



if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    train_val_figure("/home/erikj/projects/insidrug/py_proj/erikj/loggs/2020_10_30_10_36_18_fullrun/train_log.txt")