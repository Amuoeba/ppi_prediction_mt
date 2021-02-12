# General imports
import os
import json
import dash_bootstrap_components as dbc
import dash_html_components as html
# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:

def Metadata(metadata, model_path):
    children_list = []
    if metadata:
        children_list.append(html.P(f"Selected experiment: {model_path}"))
        children_list.append(html.H3("Model meta-parameters"))
        metadata = json.loads(metadata)

        longest = 0
        for x in metadata:
            if len(metadata[x]) >= longest:
                longest = len(metadata[x])

        for i, k in enumerate(metadata):
            if i % 2 == 0:
                color = "#FFC792"
            else:
                color = "#FF8848"
            values = []
            for j, v in enumerate(metadata[k]):
                if j == 0:
                    values.append(dbc.Col(f"{k}", style={"background-color": "#FFDB1D"}))
                values.append(dbc.Col(f"{v}: {metadata[k][v]}", style={"background-color": color}))
            while j < longest - 1:
                values.append(dbc.Col(f"", style={"background-color": color}))
                j += 1
            children_list.append(dbc.Row(values))

    print(children_list)
    return children_list


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
