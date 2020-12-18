# General imports
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import flask
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.components.navbar import Navbar
from visualizations.dash_app.app import app
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
    

def HomePage():
    layout = html.Div([
        Navbar(),
        "This is a home page",
        html.Video(src = "/static/project.mp4",controls=True)
        ])
    return layout




@app.server.route('/static/<file_name>')
def serve_static(file_name):
    print(f"Path to video: {file_name}")
    return flask.send_from_directory("/home/erikj/projects/insidrug/py_proj/erikj/", "project.mp4")

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")