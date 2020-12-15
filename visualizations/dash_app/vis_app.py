# General imports
import os
import glob
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Div import Div
import dash_table
from numpy.lib import utils
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import flask
import cv2
import base64


# Project specific imports

# Imports from internal libraries
from visualizations.heatmaps import protein_distogram_heatmap, train_val_figure
from mol_readers.pdb_transforms import PandasMolStructure
import config
import utils as ut


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Two chains
test_pdb_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1acb.ent.pdb"
# Sample 0
test_pdb_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1fc2.ent.pdb"
# Multiple chains
# test_pdb_file = "/home/erikj/projects/PDBind_exploration/data/v2019-other-PL/6cha/6cha_protein.pdb"
# Small ligand


# Experiment vis initialization


def get_experiments(exp_root):
    ignore = {".DS_Store", "._.DS_Store"}
    logs = [f"{exp_root}/{x}" for x in os.listdir(exp_root) if x not in ignore]
    return logs


logs = get_experiments(config.LOG_PATH)


def get_nn_vis_epochs(experiment_root):
    return os.listdir(f"{config.LOG_PATH}/{experiment_root}/nn_vis/")


def get_layers_filter_vis(experiment_root):
    return os.listdir(f"{config.LOG_PATH}/{experiment_root}/nn_vis/0/filter_viss/")


def get_samples_activation_vis(experiment_root):
    samples = os.listdir(f"{config.LOG_PATH}/{experiment_root}/nn_vis/0/")
    samples = [x for x in samples if x != "filter_viss"]
    return samples


def get_layers_activation_vis(experiment_root, sample):
    layers = os.listdir(
        f"{config.LOG_PATH}/{experiment_root}/nn_vis/0/{sample}/")
    return layers


image_directory = '/home/erikj/projects/insidrug/py_proj/erikj/loggs/2020_10_30_10_36_18_fullrun/images/'
list_of_images = [os.path.basename(x)
                  for x in glob.glob(f'{image_directory}*.png')]
static_image_route = '/static/'


app_state = {"show_het_atoms": False}

pdbStructure = PandasMolStructure()
df = pdbStructure.get_pandas_structure(
    test_pdb_file, het_atom=app_state["show_het_atoms"])
heatmap_fig = protein_distogram_heatmap(
    pdbStructure.get_pairwise_euclidean_atom(df))


app.layout = html.Div(children=[
    dcc.Tabs([
        dcc.Tab(label='Experiment results', children=[
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            ),
            html.H3("Filter weight distributions"),
            html.P("Select experiment"),
            dcc.Dropdown(id="experiment-dropdown",
                         options=[{"label": f"{x[1]}", "value": f"{x[1]}"}
                                  for x in list(map(lambda x: os.path.split(x), logs))]
                         ),
            html.P("Select layer for filter distribution visualization:"),
            dcc.Dropdown(
                id="filter-vis-layers",
                options=[],
                value=[]
            ),
            dcc.Slider(
                id='epoch-slider',
                min=0,
                max=0,
                step=1,
                value=0,
                updatemode="drag"
            ),
            html.Img(id='filter-distribution-image',
                     style={'height': '500px', 'width': '500px'}),
            html.H3("Activations"),
            dcc.Dropdown(
                id="activation-vis-samples",
                options=[],
                value=[]
            ),
            dcc.Dropdown(
                id="activation-vis-layers",
                options=[],
                value=[]
            ),
            dcc.Slider(
                id='epoch-slideract-vis',
                min=0,
                max=0,
                step=1,
                value=0,
                updatemode="drag"
            ),
            html.Div(id="activation-images"),
            html.H3("Train and validation loss"),
            html.Div(id="train-graph"),
            html.Div(id="val-graph")

        ]),
        dcc.Tab(label='Complex distogram', children=[
                html.H4(f"Displaying pdb file: {test_pdb_file}"),
                html.Button('Toggle het_atoms',
                            id='toggle_het_atoms', n_clicks=0),
                html.Div([
                    dcc.Loading(
                        id="loading-2",
                        children=[dcc.Graph(
                            id="heatmap_example",
                            figure=heatmap_fig
                        )],
                        type="circle")
                ]),
                html.Div([
                    dcc.RangeSlider(
                        id='clip_range_slider',
                        min=0,
                        max=100,
                        step=0.2,
                        marks=dict(zip(list(range(0, 100, 5)), [
                                   str(x) for x in range(0, 100, 5)])),
                        value=[0, 100]
                    ),
                    html.P(id="max_clip_feedback"),
                    html.P(id="min_clip_feedback")
                ], style={'width': '75%'}),
                html.Div(
                    [dash_table.DataTable(
                        id='hovered_residues',
                        columns=None,
                        data=None,
                    )], style={'width': '75%'}
                )
                ]),
        dcc.Tab(label='Atom proximity stats', children=[

        ])
    ])
])


@app.callback(Output('experiment-dropdown', 'options'),
              [Input('interval-component', 'n_intervals')])
def update_experiment_list(n):
    print("Updating experiments")
    logs = get_experiments(config.LOG_PATH)
    opts = [{"label": f"{x[1]}", "value": f"{x[1]}"} for x in list(map(lambda x: os.path.split(x), logs))]
    return opts


@app.callback(
    [Output('hovered_residues', 'columns'),
     Output('hovered_residues', 'data')],
    [Input('heatmap_example', 'hoverData')])
def debug_hover(hover_data):
    print("In hover")
    x = hover_data["points"][0]["y"]
    y = hover_data["points"][0]["x"]
    rows = df.iloc[[x, y]]
    cols = [{"name": i, "id": i} for i in rows.columns]
    data = rows.to_dict("rows")
    return cols, data


@app.callback(
    Output('heatmap_example', 'figure'),
    [Input('toggle_het_atoms', 'n_clicks'),
     Input('clip_range_slider', 'value')])
def togle_het_atoms(n_clicks, slider_val):
    triger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triger_id == "toggle_het_atoms":
        app_state["show_het_atoms"] = not app_state["show_het_atoms"]
    df = pdbStructure.get_pandas_structure(
        test_pdb_file, het_atom=app_state["show_het_atoms"])
    heatmap_fig = protein_distogram_heatmap(pdbStructure.get_pairwise_euclidean_atom(df),
                                            clip_min=slider_val[0], clip_max=slider_val[1])
    print(f"Clicked toggle: {app_state}")
    print(f"Data shape: {df.shape}")
    print()
    return heatmap_fig


@app.callback(
    [Output('min_clip_feedback', 'children'),
     Output('max_clip_feedback', 'children')],
    [Input('clip_range_slider', 'value')])
def update_output(value):
    max_val = f"Max clip: {value[1]}"
    min_val = f"Min clip: {value[0]}"
    return min_val, max_val

# Experiment results callbacks


@app.callback(
    [Output('epoch-slider', 'min'),
     Output('epoch-slider', 'max'),
     Output('epoch-slider', 'marks'),
     Output('epoch-slideract-vis', 'min'),
     Output('epoch-slideract-vis', 'max'),
     Output('epoch-slideract-vis', 'marks')],
    [Input('experiment-dropdown', 'value')])
def selected_eperiment(exp_root):
    print(f"experiment selected: {exp_root}")
    epochs = get_nn_vis_epochs(exp_root)
    slider_min = 0
    slider_max = len(epochs) - 1
    marks = dict(zip(list(range(slider_min, slider_max, 10)), [
                 str(x) for x in range(slider_min, slider_max, 10)]))

    return slider_min, slider_max, marks, slider_min, slider_max, marks


@app.callback(
    [Output('filter-vis-layers', 'options'),
     Output('filter-vis-layers', 'value'),
     Output('activation-vis-samples', 'options'),
     Output('activation-vis-samples', 'value'),
     Output('activation-vis-layers', 'options'),
     Output('activation-vis-layers', 'value')],
    [Input('experiment-dropdown', 'value')])
def set_available_layers(exp_root):
    layers = get_layers_filter_vis(exp_root)
    entries = [{"label": x, "value": x} for x in layers]

    samples = get_samples_activation_vis(exp_root)
    sample_entries = [{"label": x, "value": x} for x in samples]
    start_sample = samples[0]
    layers_act = get_layers_activation_vis(exp_root, start_sample)
    layers_act_entries = [{"label": x, "value": x} for x in layers_act]

    return entries, layers[0], sample_entries, samples[0], layers_act_entries, layers_act[0]


@app.callback(
    Output("filter-distribution-image", "src"),
    [Input('epoch-slider', 'value')],
    [State("experiment-dropdown", "value"),
     State("filter-vis-layers", "value")])
def update_image_epoch_selection(slider_value, selected_experiment, selected_layer):
    print(f"Slider: {slider_value}")
    print(f"Experiment: {selected_experiment}")
    print(f"layer: {selected_layer}")
    full_image_path = f"{static_image_route}{selected_experiment}/nn_vis/{slider_value}/filter_viss/{selected_layer}/weight_distributions.png"
    print(f"Fulll path: {full_image_path}")

    return full_image_path


@app.callback(
    Output("activation-images", "children"),
    [Input('epoch-slideract-vis', 'value')],
    [State("experiment-dropdown", "value"),
     State("activation-vis-samples", "value"),
     State("activation-vis-layers", "value")])
def update_activations_image(slider_value, selected_experiment, selected_sample, selected_layer):
    print(f"Slider: {slider_value}")
    print(f"Experiment: {selected_experiment}")
    print(f"layer: {selected_layer}")

    # print(f"Fulll path: {full_image_path}")

    all_images = []
    for image in os.listdir(f"{config.LOG_PATH}/{selected_experiment}/nn_vis/{slider_value}/{selected_sample}/{selected_layer}/"):
        # full_image_path = f"{static_image_route}{selected_experiment}/nn_vis/{slider_value}/{selected_sample}/{selected_layer}/{image}"
        full_image_path = f"{config.LOG_PATH}/{selected_experiment}/nn_vis/{slider_value}/{selected_sample}/{selected_layer}/{image}"
        all_images.append(html.Img(id='filter-activation-image', style={
                          'height': '500px', 'width': '500px'}, src=bae64_encoded_image(full_image_path)))
    return all_images


@app.callback(
    [Output("train-graph", "children"), Output("val-graph", "children")],
    [Input('experiment-dropdown', 'value')]
)
def update_train_and_val_graphs(selected_experiment):
    print(
        f"Selected experiment:{os.path.join(config.LOG_PATH,selected_experiment)}")
    _, train_log_name = os.path.split(ut.logger.train_log)
    train_log_path = os.path.join(
        config.LOG_PATH, selected_experiment, train_log_name)

    _, val_log_name = os.path.split(ut.logger.val_log)
    val_log_path = os.path.join(
        config.LOG_PATH, selected_experiment, val_log_name)
    print(train_log_path)
    print(val_log_path)

    train_fig = train_val_figure(train_log_path)
    val_fig = train_val_figure(val_log_path)

    return dcc.Graph(figure=train_fig), dcc.Graph(figure=val_fig)


def bae64_encoded_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    retval, buffer = cv2.imencode('.jpg', image)
    image_encoded_string = str(base64.b64encode(buffer))[2:-1]
    image_encoded_string = f"data:image/jpg;base64,{image_encoded_string}"
    return image_encoded_string


@app.server.route("/static/<experiment>/nn_vis/<epoch>/filter_viss/<layer>/weight_distributions.png")
def serve_image_1(experiment, epoch, layer):
    f_name = f"{experiment}/nn_vis/{epoch}/filter_viss/{layer}/weight_distributions.png"
    print(f"Serving image {f_name}")
    return flask.send_from_directory(config.LOG_PATH, f_name)


@app.server.route("/static/<experiment>/nn_vis/<epoch>/<sample>/<layer>/<image>.png")
def serve_image_2(experiment, epoch, sample, layer, image):
    f_name = f"{experiment}/nn_vis/{epoch}/{sample}/{layer}/{image}.png"
    print(f"Serving image {f_name}")
    image = cv2.imread(f"{config.LOG_PATH}/{f_name}")
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    retval, buffer = cv2.imencode('.jpg', image)
    image_encoded_string = str(base64.b64encode(buffer))[2:-1]
    image_encoded_string = f"data:image/jpg;base64,{image_encoded_string}"
    return image_encoded_string


# @app.server.route('{}<image_path>.png'.format(static_image_route))
# def serve_image(image_path):
#     image_name = '{}.png'.format(image_path)
#     if image_name not in list_of_images:
#         raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
#     print(f"Serving image: {image_directory}   {image_name}")
#     print(f"Im path: {image_path}")
#     return flask.send_from_directory(image_directory, image_name)

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    app.run_server(host="0.0.0.0", debug=True, port=1112)
