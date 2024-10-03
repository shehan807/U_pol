import sys, os
from openmm.app import *
from openmm import *
from simtk.unit import *
from sys import stdout

#input/output files
pdb_file = "water.pdb" 
residue_file = 'water_residue.xml'
forcefield_file = 'water.xml'

# first load bonddefinitions into Topology
Topology().loadBondDefinitions(residue_file)

# DrudeSCF integrator for optimizing Drude oscillators.
# timestep shouldn't matter, because atoms are frozen by zeroing masses in xml file.  But use a small timestep anyway
integrator = DrudeSCFIntegrator(0.00001*picoseconds)

# read in pdb file
pdb = PDBFile(pdb_file)
modeller = Modeller(pdb.topology, pdb.positions)

# create forcefield from *.xml force field file, and add extra particles (as needed) with modeller
forcefield = ForceField(forcefield_file)
modeller.addExtraParticles(forcefield)

# by default, no cutoff is used, so all interactions are computed.  This is what we want for gas phase PES...no Ewald!!
system = forcefield.createSystem(modeller.topology, constraints=None, rigidWater=True)
for i in range(system.getNumForces()):
    f = system.getForce(i)
    f.setForceGroup(i)

#  CUDA vs CPU platform?
# shouldn't need CUDA here for dimers/trimers, etc. ...
# but we find that CPU platform is unstable with DrudeSCF, for instance the current example gives NaN for Drude coordinates with CPU platform...

#platform = Platform.getPlatformByName('CPU')
platform = Platform.getPlatformByName('CUDA')
simmd = Simulation(modeller.topology, system, integrator, platform)
simmd.context.setPositions(modeller.positions)

# integrate one step to optimize Drude positions.  Note that atoms won't move if masses are set to zero
simmd.step(1)

# now call/print energy of system
state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
print("total Energy", str(state.getPotentialEnergy()))
for j in range(system.getNumForces()):
   f = system.getForce(j)
   print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

print('Done!')

exit()
