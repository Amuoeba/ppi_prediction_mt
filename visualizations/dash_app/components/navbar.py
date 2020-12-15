# General imports
import os
import dash_bootstrap_components as dbc
# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
    



def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Experiment explorer", href="/experiment-explorer")),
            dbc.NavItem(dbc.NavLink("Distogram visualization", href="/distogram-visualization")),
            # dbc.DropdownMenu(
            #     nav=True,
            #     in_navbar=True,
            #     label="Menu",
            #     children=[
            #         dbc.DropdownMenuItem("Entry 1"),
            #         dbc.DropdownMenuItem("Entry 2"),
            #         dbc.DropdownMenuItem(divider=True),
            #         dbc.DropdownMenuItem("Entry 3"),
            #     ],
            # ),
        ],
        brand="Home",
        brand_href="/home",
        sticky="top"
    )
    return navbar



if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")