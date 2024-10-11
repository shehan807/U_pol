#!/usr/bin/env python

# Import standard Python modules 
from openmm.app import *
from openmm import *
from simtk.unit import *
import numpy as np
from scipy.optimize import minimize

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
    
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    print("\nq")
    print(q)
    print("\nr")
    print(r)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

def set_constants():
    global ONE_4PI_EPS0 
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
    ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))

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

def get_inputs(openmm=True, psi4=False, scf='openmm',**kwargs):
    """
    Function to generate inputs based on OpenMM realization (i.e., 
    pdb, ff.xml, and residue.xml as inputs) or Psi4 implementation
    (i.e., mol.cif). 
    
    Arguments:
    <bool> openmm
        boolean for openmm inputs
        **kwargs: keyword arguments 
            <str> pdb 
                pdb path 
            <str> ff_xml 
                ff.xml path 
            <str> res_xml
                residue.xml path 
    <bool> psi4
        boolean for psi4 inputs
        **kwargs: keyword arguments 
            <str> cif 
                mol.cif path 
    <str> scf
        method for optimizing drude positions
        - "openmm" (used to check for accuracy)
        - "custom"
    """
    if openmm:
        # Handle input errors 
        inputs = ['pdb','ff_xml','res_xml']
        for input in inputs: 
            if input not in kwargs:
                raise ValueError(f"Missing '{input}' file for OpenMM implementation.")
        
        # use openMM to obtain bond definitions and atom/Drude positions
        Topology().loadBondDefinitions(kwargs['res_xml'])
        integrator = DrudeSCFIntegrator(0.00001*picoseconds)
        integrator.setRandomNumberSeed(123) 
        pdb = PDBFile(kwargs['pdb'])
        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField(kwargs['ff_xml'])
        modeller.addExtraParticles(forcefield)
        system = forcefield.createSystem(modeller.topology, constraints=None, rigidWater=True)
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            f.setForceGroup(i)
        platform = Platform.getPlatformByName('CUDA')
        simmd = Simulation(modeller.topology, system, integrator, platform)
        simmd.context.setPositions(modeller.positions)
        
        drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        
        positions = simmd.context.getState(getPositions=True).getPositions()
        
        # optimize drude positions using OpenMM
        simmd.step(1)
        state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
        Uind_openmm = state.getPotentialEnergy() 
        print("=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print("total Energy", str(Uind_openmm))
        for j in range(system.getNumForces()):
           f = system.getForce(j)
           print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
            
        if scf == "openmm": # if using openmm for drude opt, update positions
            positions = simmd.context.getState(getPositions=True).getPositions()
        
        numDrudes = drude.getNumParticles()
        drude_indices  = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]
        parent_indices = [drude.getParticleParameters(i)[1] for i in range(numDrudes)]
        
        # Initialize r_core, r_shell, q_core (q_shell is not needed)
        topology = modeller.getTopology()
        r_core = []; r_shell = []; q_core = []; q_shell=[]; alphas = []
        for i, res in enumerate(topology.residues()):
            res_core_pos = []; res_shell_pos = []; res_charge = []; res_shell_charge = []; res_alpha = []
            for atom in res.atoms():
                if atom.index in drude_indices:
                    continue # these are explored in parents
                charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
                charge = charge.value_in_unit(elementary_charge)
                if atom.index in parent_indices:
                    drude_index = drude_indices[parent_indices.index(atom.index)] # map parent to drude index
                    drude_pos = list(positions[drude_index])
                    drude_pos = [p.value_in_unit(nanometer) for p in drude_pos]
                    drude_params = drude.getParticleParameters(parent_indices.index(atom.index))
                    drude_charge = drude_params[5].value_in_unit(elementary_charge)
                    alpha = drude_params[6]
                    res_shell_charge.append(drude_charge)
                else:
                    res_shell_charge.append(0.0)
                    drude_pos = [0.0,0.0,0.0]
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
    elif psi4:
        pass 
        # TODO: Given initial positions of a crystal structure or trajectory file, 
        # initialize shell charge site positions and charges
        # add initial drude particles positions randomly 
        # set_drudes(...)
        # position = Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))+(unit.sum(knownPositions)/len(knownPositions))
        
    r_core = np.array(r_core)
    r_shell = np.array(r_shell)
    q_core = np.array(q_core)
    q_shell = np.array(q_shell)
    alphas = np.array(alphas)
    k = np.divide(ONE_4PI_EPS0*(q_shell)**2, alphas, where=alphas != 0, out=np.zeros_like(alphas))
    return r_core, q_core, r_shell, q_shell, k, Uind_openmm.value_in_unit(kilojoules_per_mole)


def set_drudes(r_core, weights=None):
    """ 
    TODO: Given initial positions of a crystal structure or trajectory file, 
    initialize shell charge site positions and charges

    Arguments:
    <np.array> r_core
        array of core charge site positions

    Returns:
    <np.array> r_shell
        array of all shell site positions
    """
    return 

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
    return r_core - r_shell

def Upol(d, k):
    """
    Calculates polarization energy, 
    U_pol = 1/2 Σ k_i * ||d_i||^2.

    Arguments:
    <np.array> d
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Upol
        polarization energy
    """
    d_mag = np.linalg.norm(d, axis=2)
    return 0.5 * np.sum(k * d_mag**2)

def Ucoul_vec(r_core, q_core, r_shell, q_shell):
    """
    calculates total coulomb interaction energy, 
    U_coul = ...

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np.array> d
        array of displacements between core and shell sites

    Returns:
    <np.float> U_coul
        Coulombic interaction energy
    """
    print(f"\n TESTING U_coul Vectorization")
    print(f"R_CORE:")
    print(r_core)
    print(r_core.shape)
    #print(f"R_SHELL:")
    #print(r_shell)
    #print(r_shell.shape)
    print("Rij Matrix")
    print(r_core[:, np.newaxis, :, :])
    print(r_core[np.newaxis, :, :, :])
    Rij = r_core[:, np.newaxis, :, :] - r_core[np.newaxis, :, :, :]
    print(Rij.shape) 
    print(Rij)
    dij = r_core - r_shell
    Dij = dij[:,np.newaxis,:,:]
    print(Rij - Dij)
    print(f"Q_CORE:")
    print(q_core)
    print(q_core.shape)
    Qij = q_core*q_core
    U_coul = 0.0
        # Calculate pairwise displacements between all molecules
    r_core_expanded_i = r_core[:, np.newaxis, :, :]  # Shape: (nmols, 1, natoms, 3)
    r_core_expanded_j = r_core[np.newaxis, :, :, :]  # Shape: (1, nmols, natoms, 3)
    
    # Pairwise distance vectors (Rij)
    r_core_diff = r_core_expanded_i - r_core_expanded_j  # Shape: (nmols, nmols, natoms, natoms, 3)
    rij_norms = np.linalg.norm(r_core_diff, axis=-1)  # Shape: (nmols, nmols, natoms, natoms)
    
    # Mask to exclude intramolecular interactions
    mask = np.triu(np.ones(rij_norms.shape, dtype=bool), k=1)  # Shape: (nmols, nmols, natoms, natoms)

    # Compute pairwise Coulomb core-core interactions
    U_coul_core = (q_core[:, :, np.newaxis, np.newaxis] * q_core[np.newaxis, np.newaxis, :, :]) / rij_norms  # Shape: (nmols, nmols, natoms, natoms)

    # Shell terms (similar to core-core calculation)
    r_shell_expanded_i = r_shell[:, np.newaxis, :, :]  # Shape: (nmols, 1, natoms, 3)
    r_shell_expanded_j = r_shell[np.newaxis, :, :, :]  # Shape: (1, nmols, natoms, 3)

    # Shell displacements
    di = np.where(np.linalg.norm(r_shell, axis=-1, keepdims=True) > 0, r_shell - r_core, 0.0)  # Shape: (nmols, natoms, 3)
    dj = np.where(np.linalg.norm(r_shell, axis=-1, keepdims=True) > 0, r_shell - r_core, 0.0)  # Shape: (nmols, natoms, 3)
    
    di_expanded = di[:, np.newaxis, :, :]  # Shape: (nmols, 1, natoms, 3)
    dj_expanded = dj[np.newaxis, :, :, :]  # Shape: (1, nmols, natoms, 3)
    
    # Shell terms
    U_coul_shell = (q_shell[:, :, np.newaxis, np.newaxis] * q_shell[np.newaxis, np.newaxis, :, :]) * (
        1 / np.linalg.norm(r_core_diff - dj_expanded + di_expanded, axis=-1)  # Full displacement
        - 1 / np.linalg.norm(r_core_diff - dj_expanded, axis=-1)  # Shell to core displacement
        - 1 / np.linalg.norm(r_core_diff + di_expanded, axis=-1)  # Core to shell displacement
    )

    # Total U_coul
    Ucoul_tot = np.sum((U_coul_core + U_coul_shell)[mask])
    return ONE_4PI_EPS0*U_coul 

def Ucoul(r_core, q_core, r_shell, q_shell):
    """
    calculates total coulomb interaction energy, 
    U_coul = ...

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np.array> d
        array of displacements between core and shell sites

    Returns:
    <np.float> U_coul
        Coulombic interaction energy
    """
    Ucoul_tot = 0.0
    nmols = r_core.shape[0]
    natoms = r_core.shape[1]
    shell_i = False
    shell_j = False
    for i in range(nmols):
        for j in range(i+1, nmols):
            for core_i in range(natoms):
                ri = r_core[i][core_i]
                qi = q_core[i][core_i]
                qi_shell = q_shell[i][core_i]
                if np.linalg.norm(r_shell[i][core_i]) > 0.0:
                    di = get_displacements(ri, r_shell[i][core_i])
                    shell_i = 1
                else:
                    di = 0.0
                    shell_i = 0
                for core_j in range(natoms):
                    rj = r_core[j][core_j]
                    qj = q_core[j][core_j]
                    qj_shell = q_shell[j][core_j]
                    if np.linalg.norm(r_shell[j][core_j]) > 0.0:
                        dj = get_displacements(rj, r_shell[j][core_j])
                        shell_j = 1
                    else:
                        dj = 0.0
                        shell_j = 0

                    rij = rj - ri
                    U_coul_core  = qi * qj * (1/np.linalg.norm(rij))
                    U_coul_shell = qi_shell * qj_shell * (shell_i*shell_j/np.linalg.norm(rij - dj + di))\
                                +  qi       * qj_shell * (shell_j/np.linalg.norm(rij - dj))\
                                +  qi_shell * qj       * (shell_i/np.linalg.norm(rij + di)) 
                    Ucoul_tot += U_coul_core + U_coul_shell

    return ONE_4PI_EPS0*Ucoul_tot 

def Uind(r_core, q_core, r_shell, q_shell, d, k):
    """
    calculates total induction energy, 
    U_ind = Upol + Uuu + Ustat.

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np.array> d
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uind
        induction energy
    """
    U_pol  = Upol(d, k)
    U_coul_vec = Ucoul_vec(r_core, q_core, r_shell, q_shell)
    U_coul = Ucoul(r_core, q_core, r_shell, q_shell)
    print(f"VECTORIZED U_COUL vs ORIGINAL U_COUL:\n{U_coul - U_coul_vec}")
    print("=-=-=-=-=-=-=-=-=-=-=-=-Python Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print(f"Upol={U_pol} kJ/mol\nU_coul={U_coul} kJ/mol\n")
    return U_pol + U_coul

def opt_d(d0):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    return get_displacements(r_core_opt, r_shell_opt)

def main(): 
    
    set_constants()
    
    print("WATER-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf="openmm",
                                                pdb="../benchmarks/OpenMM/water/water.pdb",
                                                ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                res_xml="../benchmarks/OpenMM/water/water_residue.xml")
    d = get_displacements(r_core, r_shell) # get initial core/shell displacements 
    U_ind = Uind(r_core, q_core, r_shell, q_shell, d, k)
    print(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
    print(f"Python U_ind = {U_ind:.4f} kJ/mol")
    print(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    
    #print("ACETONITRILE=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf="openmm",
    #                                            pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
    #                                            ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
    #                                            res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
    #d = get_displacements(r_core, r_shell) # get initial core/shell displacements 
    #U_ind = Uind(r_core, q_core, r_shell, q_shell, d, k)
    #print(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
    #print(f"Python U_ind = {U_ind:.4f} kJ/mol")
    #print(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
    #print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")



if __name__ == "__main__":
    main()
