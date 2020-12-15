# General imports
from sys import stdout
# Project specific imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
# Imports from internal libraries



if __name__ == "__main__":
    # pdb = PDBFile('/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/input.pdb')
    pdb = PDBFile('/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/argon_input.pdb')
    amber_all = "/home/erikj/anaconda3/envs/pytorch/lib/python3.7/site-packages/simtk/openmm/app/data/amber14-all.xml"
    tip3pfb = "/home/erikj/anaconda3/envs/pytorch/lib/python3.7/site-packages/simtk/openmm/app/data/amber14/tip3pfb.xml"
    forcefield = ForceField(amber_all, tip3pfb)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # context = simulation.context.getState(getVelocities=True, getForces=True,getEnergy=True).getForces(asNumpy=True)
    state = simulation.context.getState(getEnergy=True,getForces=True)
    state.getPotentialEnergy()
    print(state.getPotentialEnergy())
    print(state.getForces(asNumpy = True))
    # print(context.shape)

    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter('/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/argon_output.pdb', 10))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.step(100000)

    state = simulation.context.getState(getEnergy=True)
    state.getPotentialEnergy()

    print(state.getPotentialEnergy())