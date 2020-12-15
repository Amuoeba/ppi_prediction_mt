# General imports
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.components.navbar import Navbar
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
    

def HomePage():
    layout = html.Div([
        Navbar(),
        "This is a home page"
        ])
    return layout

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")