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

def Uuu(r_core, q, d):
    """
    Calculates electrostatic interaction energy, 
    U_uu = 1/2 Σ Σ qiqj [1/rij - 1/(rij-dj) - 1/(rij - di) + 1/(rij - dj + di)] .

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np.array> d
        array of displacements between core and shell sites

    Returns:
    <np.float> Uuu
        electrostatic interaction energy
    """
    N = r_core.shape[0]
    eps = 1e-30 # this may introduce some error
    Uuu_tot = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            #print(f"(i,j)=({i},{j})")
            #print(f"qi*qj = {q[i]}*{q[j]}")
            rij = r_core[j] - r_core[i]
            #print(f"rij={rij}")
            #print(f"di, dj={d[i]},{d[j]}")
            Uuu = q[i] * q[j] * (1/np.linalg.norm(rij) 
                                - 1/np.linalg.norm(rij - d[j])
                                - 1/np.linalg.norm(rij + d[i])
                                + 1/np.linalg.norm(rij - d[j] + d[i]))
            Uuu_tot += Uuu
    return 0.5*ONE_4PI_EPS0*Uuu 

def Ustat(r_core, r_shell, q, d):
    """
    calculates static field/induced dipole interaction energy, 
    U_stat = - Σ qi [ri*E0 - (ri + di) * E0p].

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np.array> d
        array of displacements between core and shell sites

    Returns:
    <np.float> Ustat
        field/dipole interaction energy
    """
    N = r_core.shape[0]
    E0 = np.zeros(r_core[0].shape)
    E0p = np.zeros(r_shell[0].shape)
    print(f"r_core: {r_core}")
    print(f"r_shell: {r_shell}")
    print(f"d: {d}")
    print(f"q: {q}")
    for i in range(N-1):
        j = i + 1
        E0  +=  q[j] * (r_core[j] - r_core[i]) / np.linalg.norm(r_core[j] - r_core[i])**3
        E0p += q[j] * (r_shell[j] - r_shell[i]) / np.linalg.norm(r_shell[j] - r_shell[i])**3
    print(f"E0={E0}")
    print(f"E0p={E0p}")
    #print(f"q={q}")
    #print(f"r={r}")
    #print(f"E0={E0}")
    print(f"r_core*E0 = {np.dot(r_core,E0)}")
    print(f"(r_shell+d)*E0p = {np.dot(r_shell+d,E0p)}")
    #print(f"r*E0 - (r+d)*E0p = {np.dot(r,E0)-np.dot(r+d,E0p)}")
    return -ONE_4PI_EPS0*np.sum(q * (np.dot(r_core, E0) - np.dot(r_core+d, E0p)))

def Uind(r_core, r_shell, q, d, k):
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
    <np.array> E0
        array for static field at core charge sites
    <np.array> E0p
        array for static field at shell charge site

    Returns:
    <np.float> Uind
        induction energy
    """
    U_pol  = Upol(d, k)
    U_uu   = Uuu(r_core, q, d)
    U_stat = Ustat(r_core, r_shell, q, d)
    print(f"Upol={U_pol} kJ/mol\nUuu={U_uu} kJ/mol\nUstat={U_stat} kJ/mol\n")
    return U_pol + U_uu + U_stat

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
