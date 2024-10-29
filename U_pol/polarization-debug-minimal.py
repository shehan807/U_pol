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

def set_constants():
    global ONE_4PI_EPS0 
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
    ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))

#@partial(jax.jit, static_argnames=['openmm','psi4','scf']) 

# @jit(static_argnames=['openmm','psi4','scf'])
def get_inputs(openmm=True, psi4=False, scf='openmm', jax=True, **kwargs):
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
        
    r_core  = jnp.array(r_core)
    r_shell = jnp.array(r_shell)
    q_core  = jnp.array(q_core)
    q_shell = jnp.array(q_shell)
    alphas  = jnp.array(alphas)
   
    _alphas = np.where(alphas == 0.0, 1.0, alphas)
    k = np.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)
    
    # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
    Rij = r_core[jnp.newaxis,:,jnp.newaxis,:,:] - r_core[:,jnp.newaxis,:,jnp.newaxis,:]
    
    # create Di and Dj matrices (account for nonzero values that are not true displacements)
    Dij = get_displacements(r_core, r_shell) 
    Di = Dij[:,jnp.newaxis,:,jnp.newaxis,:]
    Dj = Dij[jnp.newaxis,:,jnp.newaxis,:,:]

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:,jnp.newaxis,:,jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis,:,jnp.newaxis,:]
    Qi_core  = q_core[:,jnp.newaxis,:,jnp.newaxis]
    Qj_core  = q_core[jnp.newaxis,:,jnp.newaxis,:]
    
    return Rij, Dij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, Uind_openmm.value_in_unit(kilojoules_per_mole)

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
    
    shell_mask = jnp.linalg.norm(r_shell, axis=-1) > 0.0
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
    d_mag = jnp.linalg.norm(Dij, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

# @jit
def Ucoul(Rij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core):

    # use where solution to enable nan-friendly gradients
    Rij_norm       = jnp.linalg.norm(Rij      , axis=-1) 
    Rij_Di_norm    = jnp.linalg.norm(Rij+Di   , axis=-1)
    Rij_Dj_norm    = jnp.linalg.norm(Rij-Dj   , axis=-1)
    Rij_Di_Dj_norm = jnp.linalg.norm(Rij+Di-Dj, axis=-1)
    
    # allow divide by zero 
    _Rij_norm       = jnp.where(Rij_norm == 0.0, 1.0, Rij_norm)  
    _Rij_Di_norm    = jnp.where(Rij_Di_norm == 0.0, 1.0, Rij_Di_norm)
    _Rij_Dj_norm    = jnp.where(Rij_Dj_norm == 0.0, 1.0, Rij_Dj_norm)
    _Rij_Di_Dj_norm = jnp.where(Rij_Di_Dj_norm == 0.0, 1.0, Rij_Di_Dj_norm)

    
    U_coul = jnp.where(Rij_norm == 0.0, 0.0,         Qi_core  * Qj_core  / _Rij_norm)\
             + jnp.where(Rij_Di_norm == 0.0, 0.0,    Qi_shell * Qj_core  / _Rij_Di_norm)\
             + jnp.where(Rij_Dj_norm == 0.0, 0.0,    Qi_core  * Qj_shell / _Rij_Dj_norm)\
             + jnp.where(Rij_Di_Dj_norm == 0.0, 0.0, Qi_shell * Qj_shell / _Rij_Di_Dj_norm)

    # remove diagonal (intramolecular) components
    print(jnp.triu(U_coul))
    I = jnp.eye(U_coul.shape[0])
    U_coul = U_coul * (1 - I[:,:,jnp.newaxis,jnp.newaxis])
    print(U_coul)
    
    U_coul = 0.5 * jnp.where(jnp.isfinite(U_coul), U_coul, 0).sum() # might work in jax
    return ONE_4PI_EPS0*U_coul

# @jit
def Uind(Rij, Dij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, reshape=None):
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
    
    Dij = jax.lax.stop_gradient(Dij) # this resolves the bug where Uind is not differentiable 
    Di = jax.lax.stop_gradient(Di) # this resolves the bug where Uind is not differentiable 
    Dj = jax.lax.stop_gradient(Dj) # this resolves the bug where Uind is not differentiable

    U_pol  = Upol(Dij, k)
    U_coul = Ucoul(Rij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core)
    
    return U_pol + U_coul

def opt_d_jax(Rij, Dij0, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, methods=["BFGS"],d_ref=None, reshape=None):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    Uind_min = lambda Dij: Uind(Rij, Dij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, reshape)
     
    for method in methods:
        start = time.time()
        res = jax_minimize(Uind_min, Dij0, method=method)
        end = time.time()
        print(res)
        logger.info(f"JAX.SCIPY.OPTIMIZE Minimizer completed in {end-start:.3f} seconds!!")
        d_opt = jnp.reshape(res.x,reshape)
        if d_ref.any():
            diff = jnp.linalg.norm(d_ref-d_opt)
    return d_opt, d_opt[:,jnp.newaxis,:,jnp.newaxis,:], d_opt[jnp.newaxis,:,jnp.newaxis,:,:]
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
    testAcnit = False

    if testWater:
        logger.info("WATER-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        Rij, Dij_ref, Di_ref, Dj_ref, Qi_shell, Qj_shell, Qi_core, Qj_core, k, Uind_openmm = get_inputs(openmm=True, scf='openmm', 
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        Rij, Dij0, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, Uind_openmm = get_inputs(openmm=True, scf=None, 
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        print(Dij0)
        Dij, Di, Dj = opt_d_jax(Rij, jnp.ravel(Dij0), Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k, d_ref=Dij_ref, reshape=Dij0.shape)
        print(Dij)
        U_ind = Uind(Rij, Dij, Di, Dj, Qi_shell, Qj_shell, Qi_core, Qj_core, k)

        logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    
    #if testAcnit:
    #    logger.info("ACETONITRILE=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #    r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf="openmm",
    #                                                pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
    #                                                ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
    #                                                res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
    #    d_ref = get_displacements(r_core, r_shell) # get initial core/shell displacements 
    #    r_core, q_core, r_shell, q_shell, k, U_ind_openmm = get_inputs(OpenMM=True, scf=None,
    #                                                pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
    #                                                ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
    #                                                res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
    #    d0 = get_displacements(r_core, r_shell) # get initial core/shell displacements 
    #    d = opt_d(r_core, q_core, q_shell, d0.flatten(), k, methods=["BFGS","L-BFGS-B","TNC","SLSQP","Nelder-Mead","CG","Powell"],d_ref=d_ref)
    #    U_ind = Uind(r_core, q_core, q_shell, d, k)
    #    logger.info(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
    #    logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
    #    logger.info(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
    #    logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")



if __name__ == "__main__":
    main()
