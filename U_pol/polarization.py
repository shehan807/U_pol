#!/usr/bin/env python

# Import standard Python modules
import time
from datetime import datetime
import sys

sys.path.append(".")
from util import *
import logging
import jax.numpy as jnp
import jax
from jax import jit
from jaxopt import BFGS
from optax import safe_norm
import argparse


def set_constants():
    global ONE_4PI_EPS0
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = 1e-6 * 8.8541878128e-12 / (E_CHARGE * E_CHARGE * AVOGADRO)
    ONE_4PI_EPS0 = 1 / (4 * M_PI * EPSILON0)

@jit
def make_Sij(Rij, u_scale):
    """Build Thole screening function for intra-molecular dipole-dipole interactions."""
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    return 1.0 - (1.0 + 0.5 * Rij_norm * u_scale) * jnp.exp(-u_scale * Rij_norm)

@jit
def jnp_denominator_norm(X):
    """Enable nan-friendly gradients & divide by zero"""
    X_norm = safe_norm(X, 0.0, axis=-1)
    return jnp.where(X_norm == 0.0, jnp.inf, X_norm)

@jit
def safe_sum(X):
    """Enable safe sum for jnp matrices with infty."""
    return jnp.where(jnp.isfinite(X), X, 0).sum()

@jit
def Uself(Dij, k):
    """Calculates self energy, 1/2 Σ k_i * ||d_mag_i||^2."""
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

@jit
def Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core):
    """Compute static Coulomb energy, i.e., Q = Q_core + Q_Drude.""" 
    Rij_norm = jnp_denominator_norm(Rij)           
    U_coul_static = (Qi_core + Qi_shell) * (Qj_core + Qj_shell) / Rij_norm

    # remove intramolecular contributions
    I = jnp.eye(U_coul_static.shape[0])
    U_coul_static = U_coul_static * (1 - I[:, :, jnp.newaxis, jnp.newaxis])
    U_coul_static = 0.5 * safe_sum(U_coul_static)

    return ONE_4PI_EPS0 * U_coul_static

@jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """Compute total inter- and intra-molecular Coulomb energy.""" 
    
    # build denominator rij terms i
    Di = Dij[:, jnp.newaxis, :, jnp.newaxis, :]
    Dj = Dij[jnp.newaxis, :, jnp.newaxis, :, :]
    Rij_norm       = jnp_denominator_norm(Rij)           
    Rij_Di_norm    = jnp_denominator_norm(Rij + Di)      
    Rij_Dj_norm    = jnp_denominator_norm(Rij - Dj)      
    Rij_Di_Dj_norm = jnp_denominator_norm(Rij + Di - Dj) 

    # build Thole screening matrices
    Sij       = make_Sij(Rij, u_scale)       
    Sij_Di    = make_Sij(Rij + Di, u_scale)    
    Sij_Dj    = make_Sij(Rij - Dj, u_scale)    
    Sij_Di_Dj = make_Sij(Rij + Di - Dj, u_scale) 

    # compute intermolecular Coulomb matrix
    U_coul_inter = (
            Qi_core  * Qj_core  / Rij_norm
          + Qi_shell * Qj_core  / Rij_Di_norm
          + Qi_core  * Qj_shell / Rij_Dj_norm
          + Qi_shell * Qj_shell / Rij_Di_Dj_norm
    )
    # remove diagonal (intramolecular) components
    # NOTE: ignores ALL nonbonded interactions for bonded atoms (i.e., 1-5, 1-6, etc.)
    I = jnp.eye(U_coul_inter.shape[0])
    U_coul_inter = U_coul_inter * (1 - I[:, :, jnp.newaxis, jnp.newaxis])

    # compute intramolecular Coulomb matrix (of screened dipole-dipole pairs)
    U_coul_intra = (
            Sij       * -Qi_shell * -Qj_shell / Rij_norm
          + Sij_Di    *  Qi_shell * -Qj_shell / Rij_Di_norm
          + Sij_Dj    * -Qi_shell *  Qj_shell / Rij_Dj_norm
          + Sij_Di_Dj *  Qi_shell *  Qj_shell / Rij_Di_Dj_norm
    )
    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:, :, jnp.newaxis, jnp.newaxis]) * (
        1 - I_self[jnp.newaxis, jnp.newaxis, :, :]
    )

    U_coul_total = 0.5 * safe_sum(U_coul_inter + U_coul_intra)

    return ONE_4PI_EPS0 * U_coul_total

@jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k):
    """
    Calculate total induction energy with decomposition,
    U_total = U_induction + U_static, 
    where 
    U_induction = (U_coulomb - U_coulomb_static) + U_self.

    Arguments:
    <jaxlib.xla_extension.ArrayImp> Rij (nmol, nmol, natoms, natoms, 3)
       JAX array of core-core atom x,y,z displacements 
    <jaxlib.xla_extension.ArrayImp> Dij (nmol, natoms, 3) or (nmol*natoms*3, )
       JAX array of core-shell atom x,y,z displacements )
    <jaxlib.xla_extension.ArrayImp> Qi_shell (nmol, 1, natoms, 1)
       JAX array of shell charges, row-wise
    <jaxlib.xla_extension.ArrayImp> Qj_shell (1, nmol, 1, natoms)
       JAX array of shell charges, column-wise
    <jaxlib.xla_extension.ArrayImp> Qi_core (nmol, 1, natoms, 1)
       JAX array of core charges, row-wise
    <jaxlib.xla_extension.ArrayImp> Qj_core (1, nmol, 1, natoms)
       JAX array of core charges, column-wise
    <jaxlib.xla_extension.ArrayImp> u_scale (nmol, nmol, natoms, natoms)
       JAX array of Thole screening term, a/(alphai*alphaj)^(1/6)
    <jaxlib.xla_extension.ArrayImp> k (nmol, natoms)
       JAX array of Drude spring constants, k = q_D^2 / alpha

    Returns:
    <np.float> Uind
        induction energy
    """
    (nmol, _, natoms, _, pos) = Rij.shape
    if Dij.shape != (nmol, natoms, pos):
        Dij = jnp.reshape(Dij, (nmol, natoms, pos))

    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_coul_static = Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core)
    U_self = Uself(Dij, k)
    
    U_ind = (U_coul - U_coul_static) + U_self
    
    return U_ind


@jit
def drudeOpt(
    Rij,
    Dij0,
    Qi_shell,
    Qj_shell,
    Qi_core,
    Qj_core,
    u_scale,
    k,
    methods=["BFGS"],
    d_ref=None,
):
    """
    Iteratively determine core/shell displacements, d, by minimizing
    Uind w.r.t d.

    """
    Uind_min = lambda Dij: Uind(
        Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k
    )

    for method in methods:
        start = time.time()
        solver = BFGS(fun=Uind_min, tol=0.0001, verbose=False)
        res = solver.run(init_params=Dij0)
        end = time.time()
        logger.info(f"JAXOPT.BFGS Time: {end-start:.3f} seconds.")
        d_opt = res.params 
        try:
            if d_ref.any():
                diff = jnp.linalg.norm(d_ref - d_opt)
        except AttributeError:
            pass
    return d_opt


logger = logging.getLogger(__name__)


def main():
    #jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", True)

    global logger
    logging.basicConfig(filename="log.out", level=logging.INFO, format="%(message)s")
    logging.info(f"Log started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # logger.setLevel(logging.DEBUG)
    # logging.getLogger().setLevel(logging.DEBUG)

    set_constants()

    parser = argparse.ArgumentParser(
        description="Calculate U_ind = U_self + U_es for a selected molecule."
    )
    parser.add_argument(
        "--mol",
        type=str,
        required=True,
        choices=["water", "acnit", "imidazole", "imidazole2", "imidazole3", "pyrazine"],
        help="Molecule type (with OpenMM files).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../benchmarks/OpenMM",
        help="Directory for benchmark input files.",
    )

    args = parser.parse_args()
    dir = args.dir
    mol = args.mol

    logger.info(f"%%%%%%%%%%% STARTING {mol.upper()} U_IND CALCULATION %%%%%%%%%%%%")
    logger.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

    simmd = setup_openmm(
        pdb_file=os.path.join(dir, mol, mol + ".pdb"),
        ff_file=os.path.join(dir, mol, mol + ".xml"),
        residue_file=os.path.join(dir, mol, mol + "_residue.xml"),
    )

    start = time.time()
    Uind_openmm = U_ind_omm(simmd)
    end = time.time()
    logger.info(f"OpenMM SCF Time: {end-start:.3f} seconds.")
    
    Rij, Dij = get_Rij_Dij(simmd)
    Qi_core, Qi_shell, Qj_core, Qj_shell = get_QiQj(simmd)
    k, u_scale = get_pol_params(simmd)
    
    Dij = drudeOpt(
        Rij,
        jnp.ravel(Dij),
        Qi_shell,
        Qj_shell,
        Qi_core,
        Qj_core,
        u_scale,
        k,
    )
    
    U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
    
    print(f"U_ind type: {type(U_ind)}")
    print(f"U_ind_omm: {Uind_openmm}")
    logger.info(f"OpenMM U_ind = {Uind_openmm:.7f} kJ/mol")
    logger.info(f"Python U_ind = {U_ind:.7f} kJ/mol")
    logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
    logger.info(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
    )


if __name__ == "__main__":
    main()
