#!/usr/bin/env python

# Import standard Python modules 
import numpy as np
# import jax
# import jax.numpy as jnp
# from jax import grad, jit
from scipy.optimize import minimize

def set_constants():
    global M_PI, E_CHARGE, AVOGADRO, EPSILON0, ONE_4PI_EPS0 
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
    ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))


def get_inputs(Drude=True):
    """
    TODO: compatable function with CrystalLatte to take .cif input 
    and output arrays for core site charges and positions. For now, 
    inputs are from (a) manual or (b) MDAnalysis.
    """

    #####################RAW INPUTS########################
    # H2O Raw Positions (Initial)
    r_core = np.array(
            [[0.006599999964237213, 0.0, 0.0003000000142492354], [-0.006399999838322401, 0.0, 0.29109999537467957]]
            )
    q_core = np.array(
            [1.71636, 1.71636]
            )
    #######################################################
        
    return r_core, q_core

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

    #####################RAW INPUTS########################
    # the weights from the .xml file should give some good guess
    # for now, using what is output from OpenMM
    r_shell = np.array(
            [[0.006599999964237213, 0.0, 0.0003000000142492354], [-0.006399999838322401, 0.0, 0.29109999537467957]]
            )
    q_shell = np.array(
            [-1.71636, -1.71636]
            )
    alpha = np.array( 
                [0.000978253, 0.000978253]
            )
    #######################################################
    k = ONE_4PI_EPS0 * q_shell**2 / alpha 
    return r_shell, k

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
    d_mag = np.linalg.norm(d, axis=1)
    return 0.5 * np.sum(k * d_mag**2)

def Ucoul():
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
    #####################RAW INPUTS########################
    # H2O Raw Positions (Initial)
    r_core = np.array(
            [
                [
                [0.006599999964237213, 0.0, 0.0003000000142492354], [-0.05270000174641609, -0.07689999788999557, -0.002400000113993883], [-0.05270000174641609, 0.07689999788999557, -0.002400000113993883], [-0.017725946381688118, 8.293464803799111e-10, -0.0008074938086792827]
                ]
                , 
                [
                [-0.006399999838322401, 0.0, 0.29109999537467957],[0.07000000029802322, 0.0, 0.35109999775886536], [0.03220000118017197, 0.0, 0.20190000534057617], [0.017187558114528656, -6.582390441902102e-16, 0.28511083126068115]
                ]
            ]
            )
    q_core = np.array(
            [
                [
                1.71636, 0.55733, 0.55733, -1.11466
                ]
                , 
                [
                1.71636, 0.55733, 0.55733, -1.11466
                ]
            ]
            )
    r_shell = np.array(
            [
                [
                [0.007765778340399265, 8.520576822226844e-13, 0.0027725575491786003], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]
                ]
                , 
                [
                [-0.007186160422861576, -1.4923919738896174e-11, 0.29226040840148926], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]
                ]
            ]
            )
    #######################################################

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
                if np.linalg.norm(r_shell[i][core_i]) > 0.0:
                    di = get_displacements(ri, r_shell[i][core_i])
                    shell_i = 1
                else:
                    di = 0.0
                    shell_i = 0
                for core_j in range(natoms):
                    rj = r_core[j][core_j]
                    qj = q_core[j][core_j]
                    if np.linalg.norm(r_shell[j][core_j]) > 0.0:
                        dj = get_displacements(rj, r_shell[j][core_j])
                        shell_j = 1
                    else:
                        dj = 0.0
                        shell_j = 0
                    rij = rj - ri
                    U_coul_core  = qi * qj * (1/np.linalg.norm(rij))
                    U_coul_shell = qi * qj * (shell_i*shell_j/np.linalg.norm(rij - dj + di)
                                            - shell_j/np.linalg.norm(rij - dj)
                                            - shell_i/np.linalg.norm(rij + di)) 
                    Ucoul_tot += U_coul_core + U_coul_shell
    return ONE_4PI_EPS0*Ucoul_tot 

def Uind(r_core, r_shell, q, d, k):
    """
    calculates total induction energy, 
    U_ind = Upol + Ucoul.

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
    U_coul = Ucoul()
    print(f"Upol={U_pol} kJ/mol\nU_coul={U_coul} kJ/mol\n")
    return U_pol + U_coul 

def opt_d(d0):
    """
    TODO: Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    #####################RAW INPUTS########################
    r_core_opt = np.array(
                [[0.006600061431527138, 0.0, 0.00030016503296792507], [-0.006400051061064005, -1.1160735870709604e-15, 0.29110005497932434]]
            )
    r_shell_opt = np.array(
                [[0.007765778340399265, 8.520576822226844e-13, 0.0027725575491786003], [-0.007186160422861576, -1.4923919738896174e-11, 0.29226040840148926]]
            )
    #######################################################
    return get_displacements(r_core_opt, r_shell_opt)

def main(): 
    
    set_constants()
    r_core, q_core = get_inputs() # get positions from input file 
    r_shell, k = set_drudes(r_core) # initialize Drude positions and get spring constants (from somewhere...)
    d = get_displacements(r_core, r_shell) # get initial core/shell displacements 

    d = opt_d(d) # optimize Drude positions 
    U_ind = Uind(r_core, r_shell, q_core, d, k)
    print(U_ind) 


if __name__ == "__main__":
    main()
