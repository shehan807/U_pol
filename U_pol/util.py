#!/usr/bin/env python
from openmm.app import *
from openmm import *
from simtk.unit import *
import jax.numpy as jnp
import os
from optax import safe_norm
import numpy as np


def set_constants():
    global ONE_4PI_EPS0
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = 1e-6 * 8.8541878128e-12 / (E_CHARGE * E_CHARGE * AVOGADRO)
    ONE_4PI_EPS0 = 1 / (4 * M_PI * EPSILON0)


def get_displacements(r_core, r_shell):
    """
    Given initial positions of a crystal structure or trajectory file,
    initialize shell charge site positions and charges

    Arguments:
    <np.array> r_core
        array of core charge site positions
    <np.array> r_shell
        array of shell charge site positions

    Returns:
    <np.array> d
        array of displacements for every core/shell pair
    """
    shell_mask = safe_norm(r_shell, 0.0, axis=-1) > 0.0
    d = r_core - r_shell
    d = jnp.where(shell_mask[..., jnp.newaxis], d, 0.0)
    return d


def get_inputs(scf="openmm", **kwargs):
    """
    Function to generate inputs based on OpenMM realization (i.e.,
    pdb, ff.xml, and residue.xml as inputs).

    Arguments:
    <str> scf
        method for optimizing drude positions
        - "openmm" (used to check for accuracy)
        - "None" (see custom functions)
    **kwargs: keyword arguments
        <str> dir
            path for pdb, xml, and _residue.xml files
        <str> mol
            name of benchmark molecule
    """

    set_constants()
    logger = kwargs["logger"]
    path = kwargs["dir"]
    npz_inputs = os.path.join(path, "inputs.npz")
    # check if inputs have already been generated
    if os.path.exists(npz_inputs):
        inputs = jnp.load(npz_inputs)
        return (
            inputs["Rij"],
            inputs["Dij"],
            inputs["Qi_shell"],
            inputs["Qj_shell"],
            inputs["Qi_core"],
            inputs["Qj_core"],
            inputs["u_scale"],
            inputs["k"],
            inputs["Uind_openmm"],
        )
    else:
        if openmm:
            # Handle input errors

            pdb_file = os.path.join(path, kwargs["mol"], kwargs["mol"] + ".pdb")
            xml_file = os.path.join(path, kwargs["mol"], kwargs["mol"] + ".xml")
            residue_file = os.path.join(
                path, kwargs["mol"], kwargs["mol"] + "_residue.xml"
            )

            # use openMM to obtain bond definitions and atom/Drude positions
            Topology().loadBondDefinitions(residue_file)
            integrator = DrudeSCFIntegrator(0.00001 * picoseconds)
            integrator.setRandomNumberSeed(123)
            pdb = PDBFile(pdb_file)
            modeller = Modeller(pdb.topology, pdb.positions)
            forcefield = ForceField(xml_file)
            modeller.addExtraParticles(forcefield)
            system = forcefield.createSystem(
                modeller.topology, constraints=None, rigidWater=True
            )
            for i in range(system.getNumForces()):
                f = system.getForce(i)
                f.setForceGroup(i)
            platform = Platform.getPlatformByName("CUDA")
            simmd = Simulation(modeller.topology, system, integrator, platform)
            simmd.context.setPositions(modeller.positions)

            drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
            nonbonded = [
                f for f in system.getForces() if isinstance(f, NonbondedForce)
            ][0]

            positions = simmd.context.getState(getPositions=True).getPositions()

            # optimize drude positions using OpenMM
            simmd.step(1)
            state = simmd.context.getState(
                getEnergy=True, getForces=True, getVelocities=True, getPositions=True
            )
            Uind_openmm = state.getPotentialEnergy()
            logger.info(
                "=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
            )
            logger.info("total Energy = " + str(Uind_openmm))
            for j in range(system.getNumForces()):
                f = system.getForce(j)
                PE = str(type(f)) + str(
                    simmd.context.getState(
                        getEnergy=True, groups=2**j
                    ).getPotentialEnergy()
                )
                logger.info(PE)

            if scf == "openmm":  # if using openmm for drude opt, update positions
                positions = simmd.context.getState(getPositions=True).getPositions()

            numDrudes = drude.getNumParticles()

            drude_indices = [
                drude.getParticleParameters(i)[0] for i in range(numDrudes)
            ]
            parent_indices = [
                drude.getParticleParameters(i)[1] for i in range(numDrudes)
            ]

            # Initialize r_core, r_shell, q_core (q_shell is not needed)
            topology = modeller.getTopology()
            r_core = []
            r_shell = []
            q_core = []
            q_shell = []
            alphas = []
            tholes = []
            tholeTrue = False
            tholeMatrixMade = False
            for i, res in enumerate(topology.residues()):
                res_core_pos = []
                res_shell_pos = []
                res_charge = []
                res_shell_charge = []
                res_alpha = []
                res_thole = []
                for atom in res.atoms():
                    if atom.index in drude_indices:
                        continue  # these are explored in parents
                    charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
                    charge = charge.value_in_unit(elementary_charge)
                    if atom.index in parent_indices:
                        drude_index = drude_indices[
                            parent_indices.index(atom.index)
                        ]  # map parent to drude index
                        drude_pos = list(positions[drude_index])
                        drude_pos = [p.value_in_unit(nanometer) for p in drude_pos]
                        drude_params = drude.getParticleParameters(
                            parent_indices.index(atom.index)
                        )
                        drude_charge = drude_params[5].value_in_unit(elementary_charge)
                        alpha = drude_params[6]
                        numScreenedPairs = drude.getNumScreenedPairs()
                        if numScreenedPairs > 0:
                            tholeTrue = True
                            if not tholeMatrixMade:
                                natoms_per_res = int(
                                    (topology.getNumAtoms() - len(drude_indices))
                                    / topology.getNumResidues()
                                )
                                natoms = len(list(res.atoms()))
                                ncore = len(parent_indices)
                                nmol = len(list(topology.residues()))
                                residue_list = topology.residues()
                                tholeMatrix = np.zeros(
                                    (nmol, natoms_per_res, natoms_per_res)
                                )  # this assumes that the u_scale term is identical between core-core, shell-shell, and core-shell interactions
                                for sp_i in range(numScreenedPairs):
                                    screened_params = drude.getScreenedPairParameters(
                                        sp_i
                                    )
                                    prt0_params = drude.getParticleParameters(
                                        screened_params[0]
                                    )
                                    drude0 = prt0_params[0]
                                    core0 = prt0_params[1]
                                    alpha0 = prt0_params[6].value_in_unit(nanometer**3)
                                    imol = int(core0 / natoms)
                                    prt1_params = drude.getParticleParameters(
                                        screened_params[1]
                                    )
                                    drude1 = prt1_params[0]
                                    core1 = prt1_params[1]
                                    alpha1 = prt1_params[6].value_in_unit(nanometer**3)
                                    thole = screened_params[2]
                                    if core0 >= natoms:
                                        core0 = core0 % natoms
                                    if core1 >= natoms:
                                        core1 = core1 % natoms

                                    tholeMatrix[imol][core0][core1] = thole / (
                                        alpha0 * alpha1
                                    ) ** (1.0 / 6.0)
                                    tholeMatrix[imol][core1][core0] = thole / (
                                        alpha0 * alpha1
                                    ) ** (1.0 / 6.0)

                                tholeMatrix = list(tholeMatrix)
                                tholeMatrixMade = True
                        elif numScreenedPairs == 0:
                            tholeTrue = False
                            tholeMatrixMade = False

                        res_shell_charge.append(drude_charge)
                    else:
                        res_shell_charge.append(0.0)
                        drude_pos = [0.0, 0.0, 0.0]
                        alpha = 0.0 * nanometer**3
                    pos = list(positions[atom.index])
                    pos = [p.value_in_unit(nanometer) for p in pos]
                    alpha = alpha.value_in_unit(nanometer**3)

                    # update positions for residue
                    res_core_pos.append(pos)
                    res_shell_pos.append(drude_pos)
                    res_charge.append(charge)
                    res_alpha.append(alpha)
                r_core.append(res_core_pos)
                r_shell.append(res_shell_pos)
                q_core.append(res_charge)
                q_shell.append(res_shell_charge)
                alphas.append(res_alpha)
        r_core = jnp.array(r_core)
        r_shell = jnp.array(r_shell)
        q_core = jnp.array(q_core)
        q_shell = jnp.array(q_shell)
        alphas = jnp.array(alphas)

        _alphas = jnp.where(alphas == 0.0, jnp.inf, alphas)
        k = jnp.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)

        # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
        Rij = (
            r_core[jnp.newaxis, :, jnp.newaxis, :, :]
            - r_core[:, jnp.newaxis, :, jnp.newaxis, :]
        )

        if tholeMatrixMade:
            tholes = jnp.array(tholeMatrix)
            u_scale = (
                tholes[jnp.newaxis, ...]
                * jnp.eye(Rij.shape[0])[:, :, jnp.newaxis, jnp.newaxis]
            )
        else:
            tholes = jnp.zeros(Rij.shape[0:-2])
            u_scale = 0.0  # tholes * jnp.eye(Rij.shape[0])[:,:,jnp.newaxis,jnp.newaxis]

        # create Di and Dj matrices (account for nonzero values that are not true displacements)
        Dij = get_displacements(r_core, r_shell)

        # break up core-shell, shell-core, and shell-shell terms
        Qi_shell = q_shell[:, jnp.newaxis, :, jnp.newaxis]
        Qj_shell = q_shell[jnp.newaxis, :, jnp.newaxis, :]
        Qi_core = q_core[:, jnp.newaxis, :, jnp.newaxis]
        Qj_core = q_core[jnp.newaxis, :, jnp.newaxis, :]

        # jnp.savez(npz_inputs, Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm=Uind_openmm.value_in_unit(kilojoules_per_mole)

        return (
            Rij,
            Dij,
            Qi_shell,
            Qj_shell,
            Qi_core,
            Qj_core,
            u_scale,
            k,
            Uind_openmm.value_in_unit(kilojoules_per_mole),
        )


def get_raw_inputs(simmd, system, nonbonded_force, drude_force):
    positions = simmd.context.getState(getPositions=True).getPositions()
    r = []
    q = []
    Drude = []

    # Loop over regular particles
    for i in range(system.getNumParticles()):
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
            if parent_atom_index == i:  # If Drude particle is associated with this atom
                has_drude = True
                Drude.append(True)

        if not has_drude:
            Drude.append(False)

        q.append(charge)
        r.append(pos)
        # Output relevant information

    logger.debug(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
    )
    logger.debug("\nq")
    logger.debug(q)
    logger.debug("\nr")
    logger.debug(r)
    logger.debug(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
    )


def get_DrudeTypeMap(forcefield):
    """
    For a given openmm force field, get the 'DrudeTypeMap'
    to distinguish the name of Drude atom types.
    """
    from openmm.app.forcefield import DrudeGenerator

    drudeTypeMap = {}
    for force in forcefield._forces:
        if isinstance(force, DrudeGenerator):
            for type in force.typeMap:
                drudeTypeMap[type] = force.typeMap[type][0]
