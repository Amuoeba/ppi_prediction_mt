# General imports
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import flask
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.components.navbar import Navbar
import visualizations.dash_app.components.functions as app_f
import config

# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:


def EmbeddingVisPage():
    layout = html.Div([
        Navbar(),
        "This is embedding vis page",
        html.P("Select experiment"),
        html.Div(
            [dbc.Row(dbc.Col(html.Div("Single column"), style={"background-color": "red"})),
             dbc.Row(
                 [
                     dbc.Col(html.Div("One of three columns"), style={"background-color": "powderblue"}),
                     dbc.Col(html.Div("One of three columns")),
                     dbc.Col(html.Div("One of three columns")),
                 ]
             ),
             dbc.Row(
                 [
                     dbc.Col(
                         ["Experiments",
                          dcc.Dropdown(id="experiment-dropdown",
                                       options=[{"label": f"{x[1]}", "value": f"{x[1]}"}
                                                for x in list(map(lambda x: os.path.split(x),
                                                                  app_f.get_experiments(
                                                                      config.folder_structure_cfg.log_path)))]
                                       )]
                     ),
                     dbc.Col(
                         ["Experiment types",
                          dcc.Dropdown(id="filter-dropdown",
                                       options=[{"label": f"{x[1]}", "value": f"{x[1]}"}
                                                for x in list(map(lambda x: os.path.split(x),
                                                                  app_f.get_model_types(
                                                                      config.folder_structure_cfg.log_path)))]
                                       )]
                     )
                 ]
             )

             ],
            style={"padding": 50}
        ),
    ])
    return layout


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    print(app_f.get_experiments(config.folder_structure_cfg.log_path))
