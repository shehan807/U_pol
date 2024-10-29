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
        
    
    np_r_core  = np.array(r_core)
    np_r_shell = np.array(r_shell)
    np_q_core  = np.array(q_core)
    np_q_shell = np.array(q_shell)
    np_alphas  = np.array(alphas)
    
    # note that 
    np_k = np.true_divide(ONE_4PI_EPS0*(np_q_shell)**2, np_alphas, where=np_alphas != 0, out=np.zeros_like(np_alphas))
    
    jnp_r_core  = jnp.array(r_core)
    jnp_r_shell = jnp.array(r_shell)
    jnp_q_core  = jnp.array(q_core)
    jnp_q_shell = jnp.array(q_shell)
    jnp_alphas  = jnp.array(alphas)
    
    jnp_k = get_k(jnp_q_shell, jnp_alphas)

    print(f"%%%% Comparing harmonic spring constants %%%%%")
    print(f"numpy k = {np_k}\njax.numpy k = {jnp_k}")
    print(f"diff = {np.abs(np.array(jnp_k) - np_k)}")
    # k = jnp.where(alphas != 0, jnp.divide(ONE_4PI_EPS0 * (q_shell)**2, alphas), jnp.zeros_like(alphas))
    if jax:
        return jnp_r_core, jnp_q_core, jnp_r_shell, jnp_q_shell, jnp_k, Uind_openmm.value_in_unit(kilojoules_per_mole)
    else:
        return np_r_core, np_q_core, np_r_shell, np_q_shell, np_k, Uind_openmm.value_in_unit(kilojoules_per_mole)


# @jit
def get_k(q_shell, alphas, eps=1e-16):
    return jnp.divide(ONE_4PI_EPS0 * q_shell**2, alphas + eps)

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
    np_shell_mask = np.linalg.norm(r_shell, axis=-1) > 0.0
    np_d = r_core - r_shell
    np_d = np.where(np_shell_mask[...,jnp.newaxis], np_d, 0.0)
    
    shell_mask = jnp.linalg.norm(r_shell, axis=-1) > 0.0
    d = r_core - r_shell
    d = jnp.where(shell_mask[...,jnp.newaxis], d, 0.0)
    return d

def Upol_np(d, k):
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
    print('\nNOW DEBUGGING Upol (numpy) d_mag\n---------------------------------\n')

    d_mag = np.linalg.norm(d, axis=2)
    print(f"np d_mag = {d_mag}")
    return 0.5 * np.sum(k * d_mag**2)

# @jit
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
    print('\nNOW DEBUGGING Upol d_mag\n---------------------------------\n')

    # np_d = jax.lax.stop_gradient(d)
    # print(np_d)
    # np_d_mag = np.linalg.norm(np_d, axis=2)
    # print(f"numpy d_mag = {np_d_mag}")

    d_mag_0 = jnp.linalg.norm(d, axis=2)
    print(f"jnp d_mag = {d_mag_0}")
    
    d_mag_1 = jnp.sqrt(jnp.sum(jnp.square(d), axis=2)) # jnp.linalg.norm(d, axis=2)
    print(f"jnp d_mag = {d_mag_1}")
    
    d_mag_2 = jnp.sqrt(jnp.sum(jnp.square(d) + 1e-8, axis=2)) # jnp.linalg.norm(d, axis=2)
    print(f"jnp d_mag (with eps) = {d_mag_1}")
    
    jnp_d_mag_where = jnp.linalg.norm(jnp.where(d > 0., d, 1.), axis=2)
    print(f"jnp (+resolved where issue?) d_mag = {jnp_d_mag_where}")

    return 0.5 * jnp.sum(k * d_mag_2**2)

def Ucoul_np(r_core, q_core, q_shell, Dij):
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

    # remove diagonal (intramolecular) components 
    I = np.eye(U_coul.shape[0])
    U_coul = U_coul * (1 - I[:,:,np.newaxis,np.newaxis])
    
    # U_coul = 0.5 * jnp.ma.masked_invalid(U_coul).sum() # doesn't work in jax
    U_coul = 0.5 * np.where(np.isfinite(U_coul), U_coul, 0).sum() # might work in jax
    
    return ONE_4PI_EPS0*U_coul

# @jit
def Ucoul(r_core, q_core, q_shell, Dij):
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
    print("\n STEPPED INTO Ucoul jax\n%%%%%%%%%%%%%%%%%%%%%%%%%")
    # in jax, true_divide propogates nan values differently than numpy
    # a small epsilon (temporarily) fixes division errors 
    eps = 1e-16

    # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
    Rij = r_core[jnp.newaxis,:,jnp.newaxis,:,:] - r_core[:,jnp.newaxis,:,jnp.newaxis,:]
    
    # broadcast q_core (nmols_i, natoms_j) --> Qij (nmols_i, nmols_j, natoms_i, natoms_j)
    Qij  = q_core[jnp.newaxis,:,jnp.newaxis,:] * q_core[:,jnp.newaxis,:,jnp.newaxis]
    
    # create Di and Dj matrices (account for nonzero values that are not true displacements)
    Di = Dij[:,jnp.newaxis,:,jnp.newaxis,:]
    Dj = Dij[jnp.newaxis,:,jnp.newaxis,:,:]

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:,jnp.newaxis,:,jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis,:,jnp.newaxis,:]
    Qi_core  = q_core[:,jnp.newaxis,:,jnp.newaxis]
    Qj_core  = q_core[jnp.newaxis,:,jnp.newaxis,:]
  

    
    # print(jnp.linalg.norm(Rij,axis=-1))
    # print(jnp.linalg.norm(Rij + Di,axis=-1))
    # print(jnp.linalg.norm(Rij - Dj,axis=-1))
    # print(jnp.linalg.norm(Rij + Di - Dj,axis=-1)) 

    #Rij_y = jnp.where(jnp.linalg.norm(Rij+Di,axis=-1) == 0.0, 1.0e-16, jnp.linalg.norm(Rij+Di,axis=-1))
    #Rij_w = jnp.where(jnp.linalg.norm(Rij-Dj,axis=-1) == 0.0, 1.0e-16, jnp.linalg.norm(Rij-Dj,axis=-1))
    #Rij_x = jnp.where(jnp.linalg.norm(Rij+Di-Dj,axis=-1) == 0.0, 1.0e-16, jnp.linalg.norm(Rij+Di-Dj,axis=-1))
    #print(f"Rij_y = {Rij_y}")
    #print(f"Rij.shape = {Rij.shape}")
    #print(f"Rij_y.shape = {Rij_y.shape}")
    #print("NEW U_coul !!!")
    # print(f"Term 1:\n{Qi_core * Qj_core / jnp.linalg.norm(Rij_y,axis=-1)}")
    # print(f"Term 1:\n{jnp.where(Rij == 0.0, 1.0, Qi_core * Qj_core / jnp.linalg.norm(Rij_y,axis=-1))}")
    #U_coul = Qi_core  * Qj_core / jnp.linalg.norm(Rij,axis=-1)\
    #             + Qi_shell * Qj_core  / jnp.linalg.norm(Rij_y + Di,axis=-1)\
    #             + Qi_core  * Qj_shell / jnp.linalg.norm(Rij_y - Dj,axis=-1)\
    #             + Qi_shell * Qj_shell / jnp.linalg.norm(Rij_y + Di - Dj,axis=-1)
    #print(U_coul)
    #U_coul = jnp.where(Rij == 0.0, 0.0, U_coul)

    print("Why is term 1 not working? \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Rij = {Rij}")
    #print(f"jnp.linalg.norm(Rij,axis=-1) = {jnp.linalg.norm(Rij,axis=-1)}")
    print(f"Qi_shell*Qj_core = {Qi_shell*Qj_core}")


    #Rij_norm = jnp.linalg.norm(Rij, axis=-1)
    #y = jnp.where(Rij_norm == 0.0, 1.0, Rij_norm)
    #term1 = jnp.where(Rij_norm == 0.0, 0.0, Qi_core * Qj_core / y)
    ## term1 = jnp.where(jnp.linalg.norm(Rij, axis=-1) == 0.0, 0.0, Qi_core * Qj_core / jnp.linalg.norm(Rij, axis=-1))
    #print(f"term1 = {term1}")

    # term2 = jnp.where(jnp.linalg.norm(Rij+Di, axis=-1) == 0.0, 0.0, Qi_shell * Qj_core / Rij_y)
    #print(Di)
    #print("Rij")
    #print(Rij)
    #print("Rij+Di")
    #print(Rij+Di)
    #print("Qi_shell:")
    #print(Qi_core*Qj_shell)
    #print(Qi_shell*Qj_core)
    #print(Qi_core*Qj_core)

    
    # Rij_norm = jnp.linalg.norm(Rij, axis=-1)
    print(f"Rij.type = {type(Rij)}")
    print(f"Di.type = {type(Di)}")
    print(f"Rij.shape = {Rij.shape}")
    print(f"Di.shape = {Di.shape}")
    print((Rij + Di).shape)
    Rij_Di = jnp.add(Rij, 0.0)
    print(f"Rij_Di.type = {type(Rij_Di)}")
    print(f"Rij_Di.type = {type(Rij_Di)}")
    print(f"Rij_Di.shape = {Rij_Di.shape}")
    print(f"Rij_Di.shape = {Rij_Di.shape}")
    Rij_Di = jnp.add(Rij, Di)
    print(f"Rij_Di.shape = {Rij_Di.shape}")
    print(f"Rij_Di.shape = {Rij_Di.shape}")
    Rij_norm2 = jnp.linalg.norm(Rij_Di, axis=-1)
    y2 = jnp.where(Rij_norm2 == 0.0, 1.0, Rij_norm2)
    print(f"y2 = {y2}")
    term2 = jnp.where(Rij_norm2 == 0.0, 1.0, (Qi_shell * Qj_core) / y2)
    ##term2 = jnp.where(jnp.linalg.norm(Rij+Di, axis=-1) == 0.0, 0.0, Qi_core * Qj_core / jnp.linalg.norm(Rij+Di, axis=-1))
    print(f"term2 = {term2}")
    #term3 = jnp.where(jnp.linalg.norm(Rij-Dj, axis=-1) == 0.0, 0.0, Qi_core * Qj_core / jnp.linalg.norm(Rij-Dj, axis=-1))
    #print(f"term3 = {term3}")
    #term4 = jnp.where(jnp.linalg.norm(Rij+Di-Dj, axis=-1) == 0.0, 0.0, Qi_core * Qj_core / jnp.linalg.norm(Rij+Di-Dj, axis=-1))
    #print(f"term4 = {term4}")
    U_coul = term2  #+ term3 + term4
    #U_coul = jnp.where(jnp.linalg.norm(Rij,axis=-1) == 0.0, 0.0,\
    #               Qi_core  * Qj_core  / (Rij_y )\
    #             + Qi_shell * Qj_core  / (Rij_y + Di)\
    #             + Qi_core  * Qj_shell / (Rij_y - Dj)\
    #             + Qi_shell * Qj_shell / (Rij_y + Di - Dj))

    # remove diagonal (intramolecular) components 
    I = jnp.eye(U_coul.shape[0])
    U_coul = U_coul * (1 - I[:,:,jnp.newaxis,jnp.newaxis])
    print(f"U_coul array = {U_coul}") 
    # U_coul = 0.5 * jnp.ma.masked_invalid(U_coul).sum() # doesn't work in jax
    
    U_coul = 0.5 * jnp.sum(U_coul) # jnp.where(jnp.isfinite(U_coul), U_coul, 0).sum() # might work in jax
    print(f"U_coul value = ")
    print(f"{U_coul}")
    return ONE_4PI_EPS0*U_coul

def Uind_np(r_core, q_core, q_shell, d, k):
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
    d = d.reshape(r_core.shape) # specifically to resolve scipy.optimize handling of 1D arrays
    U_pol  = Upol_np(d, k)
    U_coul = Ucoul_np(r_core, q_core, q_shell, d)
    # print(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul}\n")
    return U_pol + U_coul

# @jit
def Uind(r_core, q_core, q_shell, d, k):
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
    print("\n STEPPED INTO Uind jax\n%%%%%%%%%%%%%%%%%%%%%%%%%")
    d = d.reshape(r_core.shape) # specifically to resolve scipy.optimize handling of 1D arrays
    print(f"d = {d}")
    U_pol  = Upol(d, k)
    U_coul = Ucoul(r_core, q_core, q_shell, d)
    # print(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul}\n")
    return U_pol + U_coul

def opt_d(r_core, q_core, q_shell, d0, k, methods=["BFGS"],d_ref=None):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    Uind_min = lambda d: Uind_np(r_core, q_core, q_shell, d, k)
    diff = 1e10
    best_method = None
    for method in methods:
        start = time.time()
        res = minimize(Uind_min, d0, method=method,options={'disp':True})
        end = time.time()
        d_opt = res.x.reshape(r_core.shape)
        print(f"Method {method} took {end-start:.4f} sec.\nOptimized d:\n{d_opt}")
        if d_ref.any():
            if np.linalg.norm(d_ref-d_opt) < diff:
                diff = np.linalg.norm(d_ref-d_opt)
                d_opt_final = d_opt
                best_method = method
            print(f"Norm Diff : {np.linalg.norm(d_ref-d_opt)}.")
    print(f"Best Method : {best_method}")
    return d_opt_final

def opt_d_jax(r_core, q_core, q_shell, d0, k, methods=["BFGS"],d_ref=None):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    Uind_min = lambda d: Uind(r_core, q_core, q_shell, d, k)
    
    best_method = None
    for method in methods:
        start = time.time()
        res = jax_minimize(Uind_min, d0, method=method)
        end = time.time()
        d_opt = res.x.reshape(r_core.shape)
        print(f"Method {method} took {end-start:.4f} sec.\nOptimized d:\n{d_opt}")
        if d_ref.any():
            diff = jnp.linalg.norm(d_ref-d_opt)
            print(f"Norm Diff : {jnp.linalg.norm(d_ref-d_opt)}.")
    return d_opt

logger = logging.getLogger(__name__)
def main(): 

    jax.config.update("jax_debug_nans", True) 
    
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
        

        start = time.time()
        jnp_r_core, jnp_q_core, jnp_r_shell, jnp_q_shell, jnp_k, U_ind_openmm = get_inputs(openmm=True, scf="openmm",
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        end = time.time()
        print(f"jnp get inputs: {end-start:.4f} seconds")

        start = time.time()
        np_r_core, np_q_core, np_r_shell, np_q_shell, np_k, U_ind_openmm = get_inputs(openmm=True, scf="openmm",jax=False,
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        end = time.time()
        print(f"np get inputs: {end-start:.4f} seconds")
        

        jnp_d_ref = get_displacements(jnp_r_core, jnp_r_shell) # get initial core/shell displacements 

        np_d_ref = get_displacements(np_r_core, np_r_shell) # get initial core/shell displacements 
        
        print(f"%%%% Comparing displacements %%%%%")
        print(f"numpy d = {np_d_ref}\njax.numpy d = {jnp_d_ref}")
        print(f"diff = {np.abs(np.array(jnp_d_ref) - np_d_ref)}")
        print("\n") 

        ### optimize np 
        np_r_core, np_q_core, np_r_shell, np_q_shell, np_k, U_ind_openmm = get_inputs(openmm=True, scf=None, jax=False,
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        np_d0 = get_displacements(np_r_core, np_r_shell) # get initial core/shell displacements 
         
        print("Optimizing numpy displacements\n-----------------------------")
        start = time.time()
        np_d = opt_d(np_r_core, np_q_core, np_q_shell, np_d0.flatten(), np_k, methods=["BFGS"],d_ref=np_d_ref)
        end = time.time()
        print(f"np opt_d: {end-start:.4f} seconds")

        ### optimize jnp 
        jnp_r_core, jnp_q_core, jnp_r_shell, jnp_q_shell, jnp_k, U_ind_openmm = get_inputs(openmm=True, scf=None,
                                                    pdb="../benchmarks/OpenMM/water/water.pdb",
                                                    ff_xml="../benchmarks/OpenMM/water/water.xml",
                                                    res_xml="../benchmarks/OpenMM/water/water_residue.xml")
        
        jnp_d0 = get_displacements(jnp_r_core, jnp_r_shell) # get initial core/shell displacements 
        #
        start = time.time()
        jnp_d = opt_d_jax(jnp_r_core, jnp_q_core, jnp_q_shell, jnp_d0.flatten(), jnp_k, methods=["BFGS"],d_ref=jnp_d_ref)
        end = time.time()
        print(f"jnp opt_d: {end-start:.4f} seconds")
        
        print(f"%%%% Comparing displacements %%%%%\n------------------------")
        print(f"numpy opt_d = {np_d}\njax.numpy opt_d = {jnp_d}")
        print(f"diff = {np.abs(np.array(jnp_d) - np_d)}")
        
        
        np_U_ind  = Uind_np(np_r_core, np_q_core, np_q_shell, np_d, np_k)
        jnp_U_ind = Uind(jnp_r_core, jnp_q_core, jnp_q_shell, jnp_d, jnp_k)

        print(f"np Uind = {np_U_ind:.3f} kJ/mol\njnp Uind = {jnp_U_ind:.3f} kJ/mol")
        
        logger.info(f"OpenMM U_ind = {U_ind_openmm:.4f} kJ/mol")
        logger.info(f"Python U_ind = {U_ind:.4f} kJ/mol")
        logger.info(f"{abs((U_ind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
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
