#!/usr/bin/env python

# Import standard Python modules 
import time
from datetime import datetime
import os, sys
sys.path.append('.')
from util import *
from openmm.app import *
from openmm import *
from simtk.unit import *
import numpy as np
from scipy.optimize import minimize
import logging
from functools import partial
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
from jax import jit
import jax 
from optax import safe_norm 

def set_constants():
    global ONE_4PI_EPS0 
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
    ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))

def get_inputs(scf='openmm', **kwargs):
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
    if openmm:
        # Handle input errors 
        path = kwargs['dir']
         
        pdb_file     = os.path.join(path, kwargs['mol'], kwargs['mol'] + ".pdb")
        xml_file     = os.path.join(path, kwargs['mol'], kwargs['mol'] + ".xml")
        residue_file = os.path.join(path, kwargs['mol'], kwargs['mol'] + "_residue.xml")

        # use openMM to obtain bond definitions and atom/Drude positions
        Topology().loadBondDefinitions(residue_file)
        integrator = DrudeSCFIntegrator(0.00001*picoseconds)
        integrator.setRandomNumberSeed(123) 
        pdb = PDBFile(pdb_file)
        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField(xml_file)
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
        logger.info("total Energy = " + str(Uind_openmm))
        for j in range(system.getNumForces()):
           f = system.getForce(j)
           PE = str(type(f)) + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy())
           logger.info(PE)
            
        if scf == "openmm": # if using openmm for drude opt, update positions
            positions = simmd.context.getState(getPositions=True).getPositions()
        
        numDrudes = drude.getNumParticles()
        drude_indices  = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]
        parent_indices = [drude.getParticleParameters(i)[1] for i in range(numDrudes)]
        
        # Initialize r_core, r_shell, q_core (q_shell is not needed)
        topology = modeller.getTopology()
        r_core = []; r_shell = []; q_core = []; q_shell=[]; alphas = []; tholes = []
        tholeTrue = False
        tholeMatrixMade = False
        for i, res in enumerate(topology.residues()):
            res_core_pos = []; res_shell_pos = []; res_charge = []; res_shell_charge = []; res_alpha = []; res_thole = []
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
                    numScreenedPairs = drude.getNumScreenedPairs()
                    if numScreenedPairs > 0:
                        tholeTrue = True
                        if not tholeMatrixMade:
                            natoms = len(list(res.atoms()))
                            ncore = len(parent_indices)
                            nmol = len(list(topology.residues()))
                            tholeMatrix = np.zeros((nmol,ncore,ncore)) # this assumes that the u_scale term is identical between core-core, shell-shell, and core-shell interactions
                            for sp_i in range(numScreenedPairs):
                                screened_params = drude.getScreenedPairParameters(sp_i)
                                prt0_params      = drude.getParticleParameters(screened_params[0])
                                drude0 = prt0_params[0]
                                core0  = prt0_params[1]
                                alpha0 = prt0_params[6].value_in_unit(nanometer**3)
                                imol = int(core0 / natoms)
                                prt1_params      = drude.getParticleParameters(screened_params[1])
                                drude1 = prt1_params[0]
                                core1  = prt1_params[1]
                                alpha1 = prt1_params[6].value_in_unit(nanometer**3)
                                thole = screened_params[2]
                                if core0 >= natoms: 
                                    core0 = (core0 % natoms) 
                                if core1 >= natoms:
                                    core1 = (core1 % natoms) 

                                tholeMatrix[imol][core0][core1] = thole / (alpha0 * alpha1)**(1./6.) 
                                tholeMatrix[imol][core1][core0] = thole / (alpha0 * alpha1)**(1./6.)

                            tholeMatrix = list(tholeMatrix)
                            tholeMatrixMade = True
                    elif numScreenedPairs == 0:
                        tholeTrue = False
                        ncore = len(parent_indices)
                        nmol = len(list(topology.residues()))
                        print(nmol, ncore)
                        tholeMatrix = list(np.zeros((nmol,ncore,ncore)))
                        
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
                print(f"atom.index = {atom.index}") 
            r_core.append(res_core_pos)
            r_shell.append(res_shell_pos)
            q_core.append(res_charge)
            q_shell.append(res_shell_charge)
            alphas.append(res_alpha)
            tholes = tholeMatrix
        
    r_core  = jnp.array(r_core)
    r_shell = jnp.array(r_shell) 
    q_core  = jnp.array(q_core)
    q_shell = jnp.array(q_shell)
    alphas  = jnp.array(alphas)
    tholes  = jnp.array(tholes)
    print("r_core, tholes")
    print(r_core, tholes)
    print(r_core.shape, tholes.shape)
    _alphas = np.where(alphas == 0.0, jnp.inf, alphas)
    k = np.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)
    
    # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
    Rij = r_core[jnp.newaxis,:,jnp.newaxis,:,:] - r_core[:,jnp.newaxis,:,jnp.newaxis,:]
    u_scale = tholes[jnp.newaxis,...] * jnp.eye(Rij.shape[0])[:,:,jnp.newaxis,jnp.newaxis] 

    # create Di and Dj matrices (account for nonzero values that are not true displacements)
    Dij = get_displacements(r_core, r_shell) 

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:,jnp.newaxis,:,jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis,:,jnp.newaxis,:]
    Qi_core  = q_core[:,jnp.newaxis,:,jnp.newaxis]
    Qj_core  = q_core[jnp.newaxis,:,jnp.newaxis,:]
    
    return Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm.value_in_unit(kilojoules_per_mole)

# @jit
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
    d = jnp.where(shell_mask[...,jnp.newaxis], d, 0.0)
    return d

# @jit
def Upol(Dij, k):
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
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

# @jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    
    Di = Dij[:,jnp.newaxis,:,jnp.newaxis,:]
    Dj = Dij[jnp.newaxis,:,jnp.newaxis,:,:]

    # use where solution to enable nan-friendly gradients
    Rij_norm       = safe_norm(Rij      , 0.0, axis=-1) 
    Rij_Di_norm    = safe_norm(Rij+Di   , 0.0, axis=-1)
    Rij_Dj_norm    = safe_norm(Rij-Dj   , 0.0, axis=-1)
    Rij_Di_Dj_norm = safe_norm(Rij+Di-Dj, 0.0, axis=-1)
    
    # allow divide by zero 
    _Rij_norm       = jnp.where(Rij_norm == 0.0,       jnp.inf, Rij_norm)  
    _Rij_Di_norm    = jnp.where(Rij_Di_norm == 0.0,    jnp.inf, Rij_Di_norm)
    _Rij_Dj_norm    = jnp.where(Rij_Dj_norm == 0.0,    jnp.inf, Rij_Dj_norm)
    _Rij_Di_Dj_norm = jnp.where(Rij_Di_Dj_norm == 0.0, jnp.inf, Rij_Di_Dj_norm)
    
    #Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    print(u_scale.shape)
    Sij = 1. - (1. + 0.5*Rij_norm*u_scale) * jnp.exp(-u_scale * Rij_norm )
    Sij_Di = 1. - (1. + 0.5*Rij_Di_norm*u_scale) * jnp.exp(-u_scale * Rij_Di_norm )
    Sij_Dj = 1. - (1. + 0.5*Rij_Dj_norm*u_scale) * jnp.exp(-u_scale * Rij_Dj_norm )
    Sij_Di_Dj = 1. - (1. + 0.5*Rij_Di_Dj_norm*u_scale) * jnp.exp(-u_scale * Rij_Di_Dj_norm )
   
    U_coul = Qi_core  * Qj_core  / _Rij_norm\
           + Qi_shell * Qj_core  / _Rij_Di_norm\
           + Qi_core  * Qj_shell / _Rij_Dj_norm\
           + Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    
    # trying with safe_norm 
    U_coul_intra = Sij       * -Qi_shell * -Qj_shell  / _Rij_norm\
                 + Sij_Di    *  Qi_shell * -Qj_shell  / _Rij_Di_norm\
                 + Sij_Dj    * -Qi_shell *  Qj_shell  / _Rij_Dj_norm\
                 + Sij_Di_Dj *  Qi_shell *  Qj_shell  / _Rij_Di_Dj_norm
    
    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self  = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:,:,jnp.newaxis,jnp.newaxis]) * (1 - I_self[jnp.newaxis,jnp.newaxis,:,:])
    U_coul_intra = 0.5 * jnp.where(jnp.isfinite(U_coul_intra), U_coul_intra, 0).sum() # might work in jax
    
    # remove diagonal (intramolecular) components
    I = jnp.eye(U_coul.shape[0])
    U_coul_inter = U_coul * (1 - I[:,:,jnp.newaxis,jnp.newaxis])
    
    U_coul = 0.5 * jnp.where(jnp.isfinite(U_coul_inter), U_coul_inter, 0).sum() # might work in jax
    return ONE_4PI_EPS0*(U_coul + U_coul_intra)

# @jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    calculates total induction energy, 
    U_ind = Upol + Uuu + Ustat.

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np is apparently a <class 'jax._src.interpreters.ad.JVPTracer'> whereas Rij is <class 'jaxlib.xla_extension.ArrayImpl'>.
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uind
        induction energy
    """
    if reshape: 
        Dij = jnp.reshape(Dij,reshape) # specifically to resolve scipy.optimize handling of 1D arrays

    U_pol  = Upol(Dij, k)
    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    logger.debug(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul}\n")
    
    return U_pol + U_coul

# @jit
def opt_d_jax(Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, methods=["BFGS"],d_ref=None, reshape=None):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    from jaxopt import BFGS, LBFGS, ScipyMinimize
    Uind_min = lambda Dij: Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape)
     
    for method in methods:
        start = time.time()
        solver = BFGS(fun=Uind_min)
        res = solver.run(init_params=Dij0)
        end = time.time()
        logger.info(f"JAXOPT.BFGS Minimizer completed in {end-start:.3f} seconds!!")
        d_opt = jnp.reshape(res.params,reshape)
        if d_ref.any():
            diff = jnp.linalg.norm(d_ref-d_opt)
    return d_opt
logger = logging.getLogger(__name__)
def main(): 

    jax.config.update("jax_debug_nans", True) 
    jax.config.update("jax_enable_x64", True) 

    global logger 
    logging.basicConfig(filename='log.out',level=logging.INFO, format='%(message)s')
    logging.info(f"Log started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # logger.setLevel(logging.DEBUG)
    # logging.getLogger().setLevel(logging.DEBUG)

    set_constants()

    testWater = True
    testAcnit = True

    if testWater:
        logger.info("WATER-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        Rij, Dij_ref, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = get_inputs(scf='openmm', dir="../benchmarks/OpenMM", mol="water")
        Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = get_inputs(scf=None, dir="../benchmarks/OpenMM", mol="water")
        
        Dij = opt_d_jax(Rij, jnp.ravel(Dij0), Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, d_ref=Dij_ref, reshape=Dij0.shape)
        
        U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)

        logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    
    if testAcnit:
        logger.info("ACETONITRILE=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        Rij, Dij_ref, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = get_inputs(scf='openmm', dir="../benchmarks/OpenMM", mol="acnit")
        Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = get_inputs(scf=None, dir="../benchmarks/OpenMM", mol="acnit")
        
        Dij = opt_d_jax(Rij, jnp.ravel(Dij0), Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, d_ref=Dij_ref, reshape=Dij0.shape)
        U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)

        logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")



if __name__ == "__main__":
    main()
