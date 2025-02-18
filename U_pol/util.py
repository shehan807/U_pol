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
            
            # ---------DEBUGGING EXCLUSIONS COMPARISON---------------
            nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
            # Add exceptions for ALL intramolecular pairs in a residue
            #for residue in modeller.getTopology().residues():
            #    atom_indices = [ atom.index for atom in residue.atoms() ]
            #    for i in range(len(atom_indices)):
            #        for j in range(i+1, len(atom_indices)):
            #            i_global = atom_indices[i]
            #            j_global = atom_indices[j]
            #            # Force the Coulomb & LJ to zero for i-j
            #            nonbonded.addException(i_global, j_global, 0.0, 1.0, 0.0, True)
            
            # Optionally turn off tail corrections as well:
            # nonbonded.setUseDispersionCorrection(False)
            # ------------------------------------------------------

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
                                logger.info(f"natoms_per_res={natoms_per_res}")
                                logger.info(f"natoms={natoms}")
                                logger.info(f"ncore={ncore}")
                                logger.info(f"nmol={nmol}")
                                logger.info(f"residue_list={residue_list}")
                                logger.info(f"tholeMatrix.shape={tholeMatrix.shape}")

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
                                    
                                    logger.info(f"screened_params={screened_params}")
                                    logger.info(f"drude0={drude0}")
                                    logger.info(f"core0={core0}")
                                    logger.info(f"alpha0={alpha0}")
                                    logger.info(f"imol={imol}")
                                    logger.info(f"prt1_params={prt1_params}")
                                    logger.info(f"drude1={drude1}")
                                    logger.info(f"core1={core1}")
                                    logger.info(f"alpha1={alpha1}")
                                    logger.info(f"thole={thole}")

                                    logger.info(f"(core0,natoms)=({core0},{natoms})")
                                    logger.info(f"(core1,natoms)=({core1},{natoms})")
                                    
                                    if core0 >= natoms:
                                        core0 = core0 % natoms
                                    if core1 >= natoms:
                                        core1 = core1 % natoms

                                    logger.info(f"(core0,natoms)=({core0},{natoms})")
                                    logger.info(f"(core1,natoms)=({core1},{natoms})")
                                    
                                    tholeMatrix[imol][core0][core1] = thole / (
                                        alpha0 * alpha1
                                    ) ** (1.0 / 6.0)
                                    tholeMatrix[imol][core1][core0] = thole / (
                                        alpha0 * alpha1
                                    ) ** (1.0 / 6.0)

                                    logger.info(f"tholeMatrix = {tholeMatrix}")
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
        

        # ============ BUILD the 4D 'bondedMask' FROM NONBONDED EXCEPTIONS =============
        # We want shape: (nmol, nmol, natoms, natoms)
        # Initialize all ones so everything is included unless we find an exception
        nmol, natoms, _ = r_core.shape
        bondedMask = jnp.ones((nmol, nmol, natoms, natoms), dtype=jnp.float64)
        
        for mol in range(nmol):
            for atom in range(natoms):
                bondedMask = bondedMask.at[mol, mol, atom, atom].set(0.0)

        # We'll parse the exceptions. Each exception is (p1, p2, chargeProd, sigma, epsilon).
        # We need to figure out which residue p1 belongs to, local index in that residue, etc.

        # A handy function to map a global atom index -> (resID, localAtomID)
        # based on your r_core ordering. We assume each residue is natoms in size:
        def map_global_index(idx):
            resID = idx // natoms   # which residue
            localID = idx % natoms
            return (resID, localID)

        # But in your code, you have the "Topology" that might place them in residue i in the order they appear.
        # For a robust approach, you can build an array global_to_resid[...] earlier. For simplicity:
        #   If each residue has 'natoms' real atoms in order, the above is enough.
        #   If your code does something else, you need the actual mapping.

        for ex_idx in range(nonbonded.getNumExceptions()):
            (p1, p2, cprod, sig, eps) = nonbonded.getExceptionParameters(ex_idx)
            # figure out which residue each belongs to
            (res1, loc1) = map_global_index(p1)
            (res2, loc2) = map_global_index(p2)

            # If cprod=0 and eps=0 => it's a full exclusion. 
            # If cprod !=0 => might be partial scale factor (like 1-4 interactions).
            # For demonstration, let's do:
            if abs(cprod.value_in_unit(elementary_charge**2)) < 1e-12 and abs(eps.value_in_unit(kilojoule_per_mole)) < 1e-12:
                # fully excluded => set mask=0
                bondedMask = bondedMask.at[res1, res2, loc1, loc2].set(0.0)
                bondedMask = bondedMask.at[res2, res1, loc2, loc1].set(0.0)
            else:
                # Example if it's 1-4 scale, you might do e.g. 0.5 
                # This depends on your forcefield. Or you can read the ratio from cprod/(q1*q2).
                # We'll just log it for demonstration:
                logger.info(f"Partial exclusion or 1-4 scale found: ex_idx={ex_idx}, p1={p1}, p2={p2}, cprod={cprod}, eps={eps}")
                # Example: set them to 0.5
                scale_factor = 0.5
                bondedMask = bondedMask.at[res1, res2, loc1, loc2].set(scale_factor)
                bondedMask = bondedMask.at[res2, res1, loc2, loc1].set(scale_factor)
            logger.info(f"bondMask={bondedMask}")

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
        logger.info("%%%%%%%%%%%%%%%%%%%%%% Printing polarization.py elements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"r_core = {r_core}")
        logger.info(f"r_core[jnp.newaxis, :, jnp.newaxis, :] = {r_core[jnp.newaxis, :, jnp.newaxis, :, :]}") 
        logger.info(f"r_core.shape={r_core.shape} -> {r_core[jnp.newaxis, :, jnp.newaxis, :].shape}; Rij.shape = {Rij.shape}\n")
        logger.info(f"Rij = {Rij}\n")

        logger.info(f"q_core = {q_core}") 
        logger.info(f"q_core[:, jnp.newaxis, :, jnp.newaxis] = {q_core[:, jnp.newaxis, :, jnp.newaxis]}") 
        logger.info(f"q_shell = {q_shell}") 
        logger.info(f"q_shell[:, jnp.newaxis, :, jnp.newaxis] = {q_shell[:, jnp.newaxis, :, jnp.newaxis]}\n")

        logger.info(f"Qi_shell = {Qi_shell}") 
        logger.info(f"Qj_shell = {Qj_shell}") 
        logger.info(f"Qi_core = {Qi_core}") 
        logger.info(f"Qj_core = {Qj_core}\n")
        logger.info(f"q_core.shape={q_core.shape} -> Qi_core.shape={Qi_core.shape} & Qj_core.shape = {Qj_core.shape}\n")


        logger.info(f"Dij = {Dij}\n")

        logger.info(f"Thole Screening and Polarization!!") 
        logger.info(f"k = {k}") 
        logger.info(f"_alphas = {_alphas}") 
        logger.info(f"tholes = {tholes}") 
        logger.info(f"u_scale = {u_scale}") 
        logger.info(f"") 
        logger.info("%%%%%%%%%%%%%%%%%%%%%% Printing polarization.py elements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

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
            bondedMask,
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
