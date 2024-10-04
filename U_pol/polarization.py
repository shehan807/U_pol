#!/usr/bin/env python

# Import standard Python modules 
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize

def set_drudes(r_core, q_shell):
    """ 
    Given initial positions of a crystal structure or trajectory file, 
    initialize shell charge site positions and charges

    Arguments:
    <np.array> r_core
        array of core charge site positions
    <np.array> q_core
        array of core charge site charges

    Returns:
    <np.array> r
        array of all core and shell site positions
    <np.array> q
        array of all core and shell site charges
    """

    r_shell = None
    q_shell = None
    return r_core + r_shell, q_core + q_shell 

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
    Upol = 0.5 * jnp.sum(k * d**2)
    return Upol 

def Uuu(r, q, d, k):
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
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uuu
        electrostatic interaction energy
    """
    N = r.shape[0]
    Uuu = 0.0

    for i in range(N):
        for i in range(i+1, N):
            rij = r[i] - r[j]
            di = None
            dj = None
            Uuu += q[i] * q[j] * (1/rij 
                                - 1/(rij - dj)
                                - 1/(rij - di)
                                + 1/(rij - dj + di))
    return Uuu 

def Ustat(r, q, d, k, E0, E0p):
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
    <np.array> k
        array of harmonic spring constants for core/shell pairs
    <np.array> E0
        array for static field at core charge sites
    <np.array> E0p
        array for static field at shell charge site

    Returns:
    <np.float> Ustat
        field/dipole interaction energy
    """
    Ustat = -jnp.sum(q * (jnp.dot(r, E0) - jnp.dot(r+d, E0p)))
    return Ustat

def Uind(r, q, d, k, E0, E0p):
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
    Uind = Upol(d, k) + Uuu(r, q, d, k) + Ustat(r, q, d, k, E0, E0p)
    return Uind

def opt_d(d0):
    """
    Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    d = None
    return d
   
def main(): 
    pass
if __name__ == "__main__":
    main()
