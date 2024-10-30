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
        r_core = []; r_shell = []; q_core = []; q_shell=[]; alphas = []; tholes = []
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
                    for n in res.atoms():
                        if n.index in drude_indices:
                            n_drude = ""
                        else:
                            n_drude = "not"
                        print(f"atom {n.index} is {n_drude} a drude")
                    for i in range(10):
                        print(i)
                        screened_params = drude.getScreenedPairParameters(i)
                        print(screened_params)
                    thole = 1.3 
                    print(f"thole = {thole}")
                    res_shell_charge.append(drude_charge)
                else:
                    res_shell_charge.append(0.0)
                    drude_pos = [0.0,0.0,0.0]
                    alpha = 0.0 * nanometer**3
                    thole = 0.0 
                pos = list(positions[atom.index])
                pos = [p.value_in_unit(nanometer) for p in pos]
                alpha = alpha.value_in_unit(nanometer**3)
                
                # update positions for residue
                res_core_pos.append(pos)
                res_shell_pos.append(drude_pos)
                res_charge.append(charge)
                res_alpha.append(alpha)
                res_thole.append(thole)
            
            r_core.append(res_core_pos)
            r_shell.append(res_shell_pos)
            q_core.append(res_charge)
            q_shell.append(res_shell_charge)
            alphas.append(res_alpha)
            tholes.append(res_thole)
    elif psi4:
        pass 
        # TODO: Given initial positions of a crystal structure or trajectory file, 
        # initialize shell charge site positions and charges
        # add initial drude particles positions randomly 
        # set_drudes(...)
        # position = Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))+(unit.sum(knownPositions)/len(knownPositions))
        
    r_core  = jnp.array(r_core)
    r_shell = jnp.array(r_shell) #* (1+1e-3) #NOTE: hard-coded but need to implement strategic initial Drude position scheme
    q_core  = jnp.array(q_core)
    q_shell = jnp.array(q_shell)
    alphas  = jnp.array(alphas)
    tholes  = jnp.array(tholes)

    _alphas = np.where(alphas == 0.0, 1.0, alphas)
    k = np.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)
    
    # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
    Rij = r_core[jnp.newaxis,:,jnp.newaxis,:,:] - r_core[:,jnp.newaxis,:,jnp.newaxis,:]
    
    # correspondingly, create an alphaij matrix and Sij matrix for thole screening
    aij_1_6 = (alphas[jnp.newaxis,:,jnp.newaxis,:] * alphas[:,jnp.newaxis,:,jnp.newaxis])**(1./6.) # nmols, nmols, natoms, natoms
    aij_1_6 = jnp.where(aij_1_6 == 0.0, jnp.inf, aij_1_6)  
    u_scale = tholes[jnp.newaxis,:,jnp.newaxis,:] / aij_1_6
    print(f"aij_1_6={aij_1_6}\ntholes={tholes}\nu_scale={u_scale}")

    #Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    #Sij = 1. - (1. + (thole * Rij_norm) / (2. * aij_1_6)) * jnp.exp(-thole * Rij_norm / aij_1_6)
    #Sij = jnp.where(Sij == 0.0, 1.0, Sij)
    print(f"Sij = {Sij}")
    # create Di and Dj matrices (account for nonzero values that are not true displacements)
    Dij = get_displacements(r_core, r_shell) 

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:,jnp.newaxis,:,jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis,:,jnp.newaxis,:]
    Qi_core  = q_core[:,jnp.newaxis,:,jnp.newaxis]
    Qj_core  = q_core[jnp.newaxis,:,jnp.newaxis,:]
    
    return Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, Uind_openmm.value_in_unit(kilojoules_per_mole)

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
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij):
    
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
   
    # trying with safe_norm 
    U_coul = jnp.where(Rij_norm == 0.0, 0.0,         Qi_core  * Qj_core  / _Rij_norm)\
             + jnp.where(Rij_Di_norm == 0.0, 0.0,    Qi_shell * Qj_core  / _Rij_Di_norm)\
             + jnp.where(Rij_Dj_norm == 0.0, 0.0,    Qi_core  * Qj_shell / _Rij_Dj_norm)\
             + jnp.where(Rij_Di_Dj_norm == 0.0, 0.0, Qi_shell * Qj_shell / _Rij_Di_Dj_norm)
    print(f"U_coul =\n {U_coul}")
    print(f"Sij =\n {Sij}")
    U_coul *= Sij 
    print(f"Sij*U_coul = {U_coul}")
    # remove diagonal (intramolecular) components
    I = jnp.eye(U_coul.shape[0])
    U_coul = U_coul * (1 - I[:,:,jnp.newaxis,jnp.newaxis])
    
    U_coul = 0.5 * jnp.where(jnp.isfinite(U_coul), U_coul, 0).sum() # might work in jax
    return ONE_4PI_EPS0*U_coul

# @jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, reshape=None):
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
    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij)
    logger.debug(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul}\n")
    
    return U_pol + U_coul

# @jit
def opt_d_jax(Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, methods=["BFGS"],d_ref=None, reshape=None):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    from jaxopt import BFGS, LBFGS, ScipyMinimize
    Uind_min = lambda Dij: Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, reshape)
     
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

    testWater = False
    testAcnit = True

    if testWater:
        logger.info("WATER-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        Rij, Dij_ref, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, Uind_openmm = get_inputs(openmm=True, scf='openmm', 
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, Uind_openmm = get_inputs(openmm=True, scf=None, 
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        Dij = opt_d_jax(Rij, jnp.ravel(Dij0), Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, d_ref=Dij_ref, reshape=Dij0.shape)
        
        U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k)

        logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    
    if testAcnit:
        logger.info("ACETONITRILE=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        Rij, Dij_ref, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, Uind_openmm = get_inputs(openmm=True, scf='openmm', 
                                                    pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
                                                    ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
                                                    res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
        
        Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, Uind_openmm = get_inputs(openmm=True, scf=None, 
                                                    pdb="../benchmarks/OpenMM/acetonitrile/acnit.pdb",
                                                    ff_xml="../benchmarks/OpenMM/acetonitrile/acnit.xml",
                                                    res_xml="../benchmarks/OpenMM/acetonitrile/acnit_residue.xml")
        
        Dij = opt_d_jax(Rij, jnp.ravel(Dij0), Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k, d_ref=Dij_ref, reshape=Dij0.shape)
        U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, Sij, k)

        logger.info(f"OpenMM U_ind = {Uind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")



if __name__ == "__main__":
    main()
