# General imports
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Project specific imports

# Imports from internal libraries
from visualizations.dash_app.pages.experiment_results import ExperimentResults
from visualizations.dash_app.pages.home import HomePage
from visualizations.dash_app.pages.distogram_vis import DistogramVisualization
from visualizations.dash_app.pages.aa_embedding_vis import EmbeddingVisPage
from visualizations.dash_app.app import app
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/experiment-explorer':
        return ExperimentResults()
    elif pathname == "/home":
        return HomePage()
    elif pathname == "/distogram-visualization":
        return DistogramVisualization()
    elif pathname == "/embedding-visualizations":
        return EmbeddingVisPage()
    else:
        return ExperimentResults()




if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
    app.run_server(host="0.0.0.0", debug=True, port=1112)
