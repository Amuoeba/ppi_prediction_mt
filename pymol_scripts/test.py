# General imports

# Project specific imports

# Imports from internal libraries


from pymol import cmd
import pymol
cmd.fragment('ala')
cmd.zoom()
cmd.png('/Users/erikjanezic/Desktop/test.png', 300, 200)
for i in cmd.get_chains():
    print(i)

