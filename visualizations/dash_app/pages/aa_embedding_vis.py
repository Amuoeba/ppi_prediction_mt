# General imports
import os
import dacite
import dataclasses as dc
from itertools import product
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import json
import flask
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.app import app
from visualizations.dash_app.components.navbar import Navbar
from visualizations.dash_app.components.metadata_view import Metadata
import visualizations.dash_app.components.functions as app_f
import utils
import config

from models import aa_embeder

# Typing imports
from typing import TYPE_CHECKING

# if TYPE_CHECKING:


# State variables

COLOR_BY = "all"


def path_to_dropdown_option(p, full_path=False):
    p = Path(p)
    if full_path:
        dropdown_option = {"label": f"{p.name}", "value": f"{p}"}
    else:
        dropdown_option = {"label": f"{p.name}", "value": f"{p.name}"}
    return dropdown_option


def generate_embeding_plot_data(model_state, selected_experiment, ):
    # Init loger and encoder
    aux_loger = utils.TrainLogger(config.folder_structure_cfg.log_path, selected_experiment)
    metadata = json.loads(aux_loger.get_experiment_metadata())
    metadata = dacite.from_dict(aa_embeder.MetaParams, metadata)
    # Metadata variables
    vocab_size = metadata.model_params.vocab_size

    print(metadata)
    encoder = aa_embeder.AA_OneHotEncoder(metadata.dataset_params.ntuple_size)

    # Init embeder model
    model = aa_embeder.NGramLanguageModeler(**dc.asdict(metadata.model_params))
    model.load_state_dict(torch.load(model_state, map_location=torch.device('cpu')))
    model.eval()

    # Generate embedings
    decoded = encoder.decode([x for x in range(vocab_size)], mode="df")

    lookup = torch.tensor([x for x in range(vocab_size)], dtype=torch.long)
    embeddings = model.embeddings(lookup)
    embeddings = embeddings.detach().numpy()

    tsne_transformed = TSNE(n_components=2).fit_transform(embeddings)

    aa_df = pd.read_csv("/home/erik/Projects/master_thesis/ppi_prediction_mt/data/aminoacids.csv")

    dropcolumns = ["FullName", "ISO1"]
    decoded_with_meta = pd.merge(left=decoded, right=aa_df, left_on="aa_0", right_on="ISO3", how="left")
    decoded_with_meta = pd.merge(left=decoded_with_meta,
                                 right=aa_df[["ISO3", "molecular_weight", "type", "aromatic"]],
                                 left_on="aa_1", right_on="ISO3", how="left", suffixes=["_0", "_1"])
    decoded_with_meta.drop(columns=dropcolumns, inplace=True)

    decoded_with_meta["enc_0"], decoded_with_meta["enc_1"] = tsne_transformed[:, 0], tsne_transformed[:, 1]
    return decoded_with_meta


def generate_embedding_graph(plotdf: pd.DataFrame, mode, color_by=all):
    n_aas = len(plotdf.filter(regex="aa.*").columns)
    colorings = list(range(n_aas)) + "all"
    modes = ["aa_type", "aa_aromatic"]
    assert color_by in colorings, f"color_by must be one of hte following{colorings}." \
                                  f"This represents how many subsequent amionacids are used to generate embeddings"
    assert mode in modes, f"mode must be one of the folllowing: {modes}"

    # Custom color maps
    features = plotdf["type_0"].unique()
    rows = list(zip(features, ["#fcba03", "#fc0303", "#1e8255", "#036bfc", "#b82abf"]))
    colormap_df = pd.DataFrame(rows, columns=["feature", "color"])
    aromatic_color_map = pd.DataFrame([[0, "#f24500"], [1, "#008549"]], columns=["feature", "color"])

    plot_df = pd.merge(left=plotdf, right=colormap_df, left_on="type_0", right_on="feature", how="left")


def aa_type_graph(data: pd.DataFrame, color_by):
    n_aas = len(data.filter(regex="aa.*").columns)
    colorings = list(range(n_aas)) + ["all"]
    assert color_by in colorings, f"color_by must be one of hte following{colorings}." \
                                  f"This represents how many subsequent amionacids are used to generate embeddings"

    # Generate features
    if color_by == "all":
        features = np.array([])
        for i in range(n_aas):
            features = np.concatenate((features, data[f"type_{i}"].unique()), 0)
        features = np.unique(features)
        features = list(product(features, repeat=n_aas))
    else:
        features = data[f"type_{color_by}"].unique()

    # Generate markers
    def generate_markers(feature):
        if color_by != "all":
            rows = list(zip(features, ["#fcba03", "#fc0303", "#1e8255", "#036bfc", "#b82abf"][:len(features)]))
            colormap_df = pd.DataFrame(rows, columns=["feature", "color"])
            markers = dict(color=colormap_df[colormap_df.feature == feature]["color"].values[0])
            return markers
        else:
            return None

    # Generate filtered encodings
    def filter_encodings_by_features(feature):
        if color_by == "all":
            mask = None
            for i, f in enumerate(feature):
                if mask is None:
                    mask = data[f"type_{i}"] == feature[i]
                else:
                    mask = mask & (data[f"type_{i}"] == feature[i])
            filtered_data = data[mask]
        else:
            filtered_data = data[(data[f"type_{color_by}"] == feature)]

        return filtered_data

    fig = go.Figure()
    for f in features:
        filtered_encodings = filter_encodings_by_features(f)
        fig.add_trace(go.Scatter(
            x=filtered_encodings["enc_0"],
            y=filtered_encodings["enc_1"],
            marker=generate_markers(f),
            mode="markers",
            showlegend=True,
            name=str(f)
        ))

    fig.update_layout(
        width=1200,
        height=800
    )

    return dcc.Graph(figure=fig), colorings


def EmbeddingVisPage():
    layout = html.Div([
        Navbar(),
        "This is embedding vis page",
        html.P("Select experiment"),
        html.Div(
            [dbc.Row([
                dbc.Col(
                    ["Experiment types",
                     dcc.Dropdown(id="filter-dropdown",
                                  options=[{"label": f"{x[1]}", "value": f"{x[1]}"}
                                           for x in list(map(lambda x: os.path.split(x),
                                                             app_f.get_model_types(
                                                                 config.folder_structure_cfg.log_path)))]
                                  )]
                ),
                dbc.Col(
                    ["Experiments",
                     dcc.Dropdown(id="EMB_experiment-dropdown",
                                  options=list(map(path_to_dropdown_option,
                                                   app_f.get_experiments(config.folder_structure_cfg.log_path))),
                                  # options=[{"label": f"{x[1]}", "value": f"{x[1]}"}
                                  #          for x in list(map(lambda x: os.path.split(x),
                                  #                            app_f.get_experiments(
                                  #                                config.folder_structure_cfg.log_path)))]
                                  )]
                ),
                dbc.Col(
                    ["Model states",
                     dcc.Dropdown(id="model-states",
                                  options=[])]
                ),
            ]
            )

            ],
            style={"padding": 50}
        ),
        html.Div(id="metadata", style={"padding": 50}),
        html.Div(id="graph-options"),
        html.Div(id="embedding-grapphs", style={"padding": 50})

    ])
    return layout


@app.callback(Output("EMB_experiment-dropdown", "options"),
              [Input("filter-dropdown", "value")],
              )
def filter_experiments(model_type):
    experiments = app_f.get_experiments(config.folder_structure_cfg.log_path)
    filtered_experiments = []
    for e in experiments:
        e = Path(e)
        aux_loger = utils.TrainLogger(config.folder_structure_cfg.log_path, e.name)
        t = aux_loger.get_experiment_type()
        if t == model_type:
            filtered_experiments.append(e)

    filtered_experiments = list(map(path_to_dropdown_option, filtered_experiments))
    return filtered_experiments


@app.callback([Output("metadata", "children"), Output("model-states", "options")],
              [Input("EMB_experiment-dropdown", "value")])
def select_expperiment(experiment):
    aux_loger = utils.TrainLogger(config.folder_structure_cfg.log_path, experiment)
    metadata = aux_loger.get_experiment_metadata()

    aux_partial = partial(path_to_dropdown_option, full_path=True)
    model_states = list(map(aux_partial, aux_loger.get_model_states()))

    return Metadata(metadata, aux_loger.current_path), model_states


# TODO figure out how to create callback on dinamically created elements
@app.callback([Output("embedding-grapphs", "children"),
               Output("graph-options", "children")],
              [Input("model-states", "value")],
              [State("EMB_experiment-dropdown", "value")])
def select_model_state(model_state, selected_experiment):
    print(f"Model state: {model_state}")
    print(f"Selected experiment: {selected_experiment}")

    plot_df = generate_embeding_plot_data(model_state, selected_experiment)

    graph, colorings = aa_type_graph(plot_df, 0)

    options = [{"label": f"{c}", "value": f"{c}"} for c in colorings]
    options = dcc.Dropdown(id="aa_graph_options", options=options)

    return graph, options


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    model_state = "/home/erik/Projects/master_thesis/data/logs/2021_02_17_03_02_59_latest/models/aa_embedder.pt"
    selected_experiment = "2021_02_17_03_02_59_latest"
    plot_data = generate_embeding_plot_data(model_state, selected_experiment)

    aa_type_graph(plot_data, "all")
