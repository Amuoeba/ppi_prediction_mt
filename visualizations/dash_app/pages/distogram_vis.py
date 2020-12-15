# General imports
import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.app import app
from visualizations.dash_app.components.navbar import Navbar
from visualizations.heatmaps import protein_distogram_heatmap, train_val_figure
from mol_readers.pdb_transforms import PandasMolStructure
import config
import utils as ut
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:


test_pdb_file = "/home/erikj/projects/PDBind_exploration/data/pdbbind_v2019_PP/PP/1acb.ent.pdb"
pdbStructure = PandasMolStructure()
app_state = {"show_het_atoms": False}
df = pdbStructure.get_pandas_structure(
    test_pdb_file, het_atom=app_state["show_het_atoms"])
heatmap_fig = protein_distogram_heatmap(
    pdbStructure.get_pairwise_euclidean_atom(df))

def DistogramVisualization():

    layout = html.Div([
        Navbar(),
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
    ])

    return layout

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

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
