#!/usr/bin/env python

# Import standard Python modules 
import os, sys
sys.path.append('.')
from util import *
from openmm.app import *
from openmm import *
from simtk.unit import *
import numpy as np
from scipy.optimize import minimize
import logging
import time

def set_constants():
    global ONE_4PI_EPS0 
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
    ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))


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
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        logger.info("total Energy" + str(Uind_openmm))
        for j in range(system.getNumForces()):
           f = system.getForce(j)
           PE = str(type(f)) + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy())
           print(PE)
           logger.info(PE)
            
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
    
    # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
    Rij = r_core[np.newaxis,:,np.newaxis,:,:] - r_core[:,np.newaxis,:,np.newaxis,:]
    
    # broadcast q_core (nmols_i, natoms_j) --> Qij (nmols_i, nmols_j, natoms_i, natoms_j)
    Qij  = q_core[np.newaxis,:,np.newaxis,:] * q_core[:,np.newaxis,:,np.newaxis]
    
    # create Di and Dj matrices (account for nonzero values that are not true displacements)
    shell_mask = np.linalg.norm(r_shell, axis=-1) > 0.0
    Dij = r_core - r_shell
    Dij = np.where(shell_mask[...,np.newaxis], Dij, 0.0)
    Di_nb = Dij
    Di = Dij[:,np.newaxis,:,np.newaxis,:]
    Dj = Dij[np.newaxis,:,np.newaxis,:,:]

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:,np.newaxis,:,np.newaxis]
    Qj_shell = q_shell[np.newaxis,:,np.newaxis,:]
    Qi_core  = q_core[:,np.newaxis,:,np.newaxis]
    Qj_core  = q_core[np.newaxis,:,np.newaxis,:]
    
    U_coul =       Qi_core  * Qj_core  / np.linalg.norm(Rij,axis=-1)\
                 + Qi_shell * Qj_core  / np.linalg.norm(Rij + Di,axis=-1)\
                 + Qi_core  * Qj_shell / np.linalg.norm(Rij - Dj,axis=-1)\
                 + Qi_shell * Qj_shell / np.linalg.norm(Rij + Di - Dj,axis=-1)       

    # exclude double counting
    I = np.eye(U_coul.shape[0])
    U_coul = U_coul * (1 - I[:,:,np.newaxis,np.newaxis])
    U_coul = np.ma.masked_invalid(U_coul).sum()
    
    return ONE_4PI_EPS0*0.5*U_coul 

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
                    Ucoul_tot += U_coul_core  + U_coul_shell

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
    start = time.time()
    U_coul_vec = Ucoul_vec(r_core, q_core, r_shell, q_shell)
    end = time.time()
    vec_time = end-start
    start = time.time()
    U_coul = Ucoul(r_core, q_core, r_shell, q_shell)
    end = time.time()
    forloop_time = end - start
    logger.debug(f"VECTORIZED U_COUL ({U_coul_vec}) vs ORIGINAL U_COUL ({U_coul}):delta\n{U_coul - U_coul_vec}")
    logger.debug(f"VECTORIZED ({vec_time:0.8f}) sec;\nFOR LOOPS  ({forloop_time:0.8f}) sec\n{forloop_time/vec_time:0.2f}x SPEEDUP")
    logger.debug("=-=-=-=-=-=-=-=-=-=-=-=-Python Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    logger.debug(f"Upol={U_pol} kJ/mol\nU_coul={U_coul} kJ/mol\n")
    return U_pol + U_coul_vec

def opt_d(d0):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    return get_displacements(r_core_opt, r_shell_opt)

logger = logging.getLogger(__name__)
def main(): 
   
    global logger 
    logging.basicConfig(filename='log.out',level=logging.DEBUG, format='%(message)s')
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    set_constants()

    testWater = True
    testAcnit = True

    if testWater:
        logger.info("WATER-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf="openmm",
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        d = get_displacements(r_core, r_shell) # get initial core/shell displacements 
        U_ind = Uind(r_core, q_core, r_shell, q_shell, d, k)
        logger.info(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    
    if testAcnit:
        logger.info("ACETONITRILE=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf="openmm",
                                                    pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
                                                    ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
                                                    res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
        d = get_displacements(r_core, r_shell) # get initial core/shell displacements 
        U_ind = Uind(r_core, q_core, r_shell, q_shell, d, k)
        logger.info(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")



if __name__ == "__main__":
    main()
