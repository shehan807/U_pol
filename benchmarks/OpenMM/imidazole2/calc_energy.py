from openmm.app import *
from openmm import *
from simtk.unit import *


def get_raw_inputs(simmd, system, nonbonded_force, drude_force):
    positions = simmd.context.getState(getPositions=True).getPositions()
    r = []
    q = []
    Drude = []

    # Loop over regular particles
    for i in range(system.getNumParticles()):
        # print(f"ATOM {i} INFO:\n")
        # Get charge, sigma, epsilon for each atom
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        charge = charge.value_in_unit(elementary_charge)

        # Get position of the atom
        pos = list(positions[i])
        pos = [p.value_in_unit(nanometer) for p in pos]

        # Check if this atom has an associated Drude particle
        has_drude = False
        for j in range(drude_force.getNumParticles()):
            # Retrieve Drude particle parameters
            params = drude_force.getParticleParameters(j)
            parent_atom_index = params[0]

            polarizability = params[6]
            print(params)
            if parent_atom_index == i:  # If Drude particle is associated with this atom
                has_drude = True
                print(
                    f" Drude Parameters for Atom {i}: Charge = {charge}, Polarizability = {polarizability}"
                )
                Drude.append(True)

        if not has_drude:
            # print(f"No Drude parameters for Atom {i}")
            Drude.append(False)

        q.append(charge)
        r.append(pos)
        # Output relevant information
        # print(f"Atom {i}: Charge = {charge}, Sigma = {sigma}, Epsilon = {epsilon}, Position = {pos}, Drude = {Drude[i]}")

    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    print("\nq")
    print(q)
    print("\nr")
    print(r)
    print("\nDrude?")
    print(Drude)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")


# input/output files
pdb_file = "imidazole.pdb"
residue_file = "imidazole_residue.xml"
forcefield_file = "imidazole.xml"

# first load bonddefinitions into Topology
Topology().loadBondDefinitions(residue_file)

# DrudeSCF integrator for optimizing Drude oscillators.
# timestep shouldn't matter, because atoms are frozen by zeroing masses in xml file.  But use a small timestep anyway
integrator = DrudeSCFIntegrator(0.00001 * picoseconds)
integrator.setRandomNumberSeed(123)

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

# platform = Platform.getPlatformByName('CPU')
platform = Platform.getPlatformByName("CUDA")
simmd = Simulation(modeller.topology, system, integrator, platform)
simmd.context.setPositions(modeller.positions)

# integrate one step to optimize Drude positions.  Note that atoms won't move if masses are set to zero
# Get the NonbondedForce which contains the charges and other parameters
# Get the NonbondedForce which contains the charges and other parameters
nonbonded_force = None
drude_force = None

# Loop through system forces to find NonbondedForce and DrudeForce
for i in range(system.getNumForces()):
    force = system.getForce(i)
    if isinstance(force, NonbondedForce):
        nonbonded_force = force
    elif isinstance(force, DrudeForce):
        drude_force = force
print("")
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
print(" Printing Positions (before Drude Optimization)                        \n")
get_raw_inputs(simmd, system, nonbonded_force, drude_force)
# simmd.step(1)
print(" Printing Positions (after Drude Optimization)                         \n")
get_raw_inputs(simmd, system, nonbonded_force, drude_force)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

# now call/print energy of system
state = simmd.context.getState(
    getEnergy=True, getForces=True, getVelocities=True, getPositions=True
)
print("total Energy", str(state.getPotentialEnergy()))
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(
        type(f),
        str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()),
    )

print("Done!")

exit()
