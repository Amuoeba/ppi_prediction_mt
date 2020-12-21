# General imports
import os
import sqlite3
# Project specific imports

# Imports from internal libraries
from mol_readers.pdbind import PDBindDataset, PandasMolStructure
import feature_constructors.categorical as catFeatures
import config
# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")

    sql_db = PDBindDataset(config.PDBIND_SQLITE_DB)
    samples = sql_db.get_all_pdbs()

    pdb_struct = PandasMolStructure()
    protein_structure = pdb_struct.get_pandas_structure(samples["protein"][0])
    seq = pdb_struct.get_protein_sequence()
    
    


    a = 1
