# General imports
import os
import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING

# if TYPE_CHECKING:


app = dash.Dash()

app.config.suppress_callback_exceptions = True



app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
