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


# @jit
def Uself(Dij, k):
    """
    Calculates self energy,
    U_self = 1/2 Σ k_i * ||d_mag_i||^2.

    Arguments:
    <np.array> Dij
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uself
        polarization energy
    """
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

# @jit
def Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core):

    # use where solution to enable nan-friendly gradients
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)

    # allow divide by zero
    _Rij_norm = jnp.where(Rij_norm == 0.0, jnp.inf, Rij_norm)

    U_coul_static = (
        (Qi_core + Qi_shell) * (Qj_core + Qj_shell) / _Rij_norm
    )
    
    # remove intramolecular contributions
    I = jnp.eye(U_coul_static.shape[0])
    mask = (1 - I[:, :, jnp.newaxis, jnp.newaxis])
    U_coul_static = U_coul_static * mask
    U_coul_static = (
        0.5 * jnp.where(jnp.isfinite(U_coul_static), U_coul_static, 0).sum()
    )  # might work in jax

    return ONE_4PI_EPS0 * (U_coul_static)

# @jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    Di = Dij[:, jnp.newaxis, :, jnp.newaxis, :]
    Dj = Dij[jnp.newaxis, :, jnp.newaxis, :, :]

    # use where solution to enable nan-friendly gradients
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    Rij_Di_norm = safe_norm(Rij + Di, 0.0, axis=-1)
    Rij_Dj_norm = safe_norm(Rij - Dj, 0.0, axis=-1)
    Rij_Di_Dj_norm = safe_norm(Rij + Di - Dj, 0.0, axis=-1)

    # allow divide by zero
    _Rij_norm = jnp.where(Rij_norm == 0.0, jnp.inf, Rij_norm)
    _Rij_Di_norm = jnp.where(Rij_Di_norm == 0.0, jnp.inf, Rij_Di_norm)
    _Rij_Dj_norm = jnp.where(Rij_Dj_norm == 0.0, jnp.inf, Rij_Dj_norm)
    _Rij_Di_Dj_norm = jnp.where(Rij_Di_Dj_norm == 0.0, jnp.inf, Rij_Di_Dj_norm)

    # Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    Sij = 1.0 - (1.0 + 0.5 * Rij_norm * u_scale) * jnp.exp(-u_scale * Rij_norm)
    Sij_Di = 1.0 - (1.0 + 0.5 * Rij_Di_norm * u_scale) * jnp.exp(-u_scale * Rij_Di_norm)
    Sij_Dj = 1.0 - (1.0 + 0.5 * Rij_Dj_norm * u_scale) * jnp.exp(-u_scale * Rij_Dj_norm)
    Sij_Di_Dj = 1.0 - (1.0 + 0.5 * Rij_Di_Dj_norm * u_scale) * jnp.exp(
        -u_scale * Rij_Di_Dj_norm
    )

    # total coulomb energy
    U_coul = (
        Qi_core * Qj_core / _Rij_norm
        + Qi_shell * Qj_core / _Rij_Di_norm
        + Qi_core * Qj_shell / _Rij_Dj_norm
        + Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    )

    # trying with safe_norm
    U_coul_intra = (
        Sij * -Qi_shell * -Qj_shell / _Rij_norm
        + Sij_Di * Qi_shell * -Qj_shell / _Rij_Di_norm
        + Sij_Dj * -Qi_shell * Qj_shell / _Rij_Dj_norm
        + Sij_Di_Dj * Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    )

    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:, :, jnp.newaxis, jnp.newaxis]) * (
        1 - I_self[jnp.newaxis, jnp.newaxis, :, :]
    )
    U_coul_intra = (
        0.5 * jnp.where(jnp.isfinite(U_coul_intra), U_coul_intra, 0).sum()
    )  # might work in jax
    
    # remove diagonal (intramolecular) components
    # note, this ignores ALL nonbonded interactions for 
    # bonded atoms (i.e., 1-5, 1-6, etc.)
    I = jnp.eye(U_coul.shape[0])
    mask = (1 - I[:, :, jnp.newaxis, jnp.newaxis])
    U_coul_inter = U_coul * mask
    U_coul_inter = (
        0.5 * jnp.where(jnp.isfinite(U_coul_inter), U_coul_inter, 0).sum()
    )  # might work in jax
    
    return ONE_4PI_EPS0 * (U_coul_inter + U_coul_intra)


# @jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    calculates total induction energy,
    U_ind = Uself + Uuu + Ustat.

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
        Dij = jnp.reshape(
            Dij, reshape
        )  # specifically to resolve scipy.optimize handling of 1D arrays

    U_self = Uself(Dij, k)
    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_coul_static = Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core)
    U_ind = U_coul - U_coul_static + U_self
    logger.info(f"U_self = {U_self} kJ/mol\nU_coul = {U_coul} kJ/mol\n")
    logger.info(f"U_coul_static = {U_coul_static} kJ/mol\nU_ind = {U_ind} kJ/mol\n")

    return U_self + U_coul


# @jit
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
    reshape=None,
):
    """
    Iteratively determine core/shell displacements, d, by minimizing
    Uind w.r.t d.

    """

    Uind_min = lambda Dij: Uind(
        Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape
    )

    for method in methods:
        start = time.time()
        solver = BFGS(fun=Uind_min, tol=0.0001)
        res = solver.run(init_params=Dij0)
        end = time.time()
        logger.info(f"JAXOPT.BFGS Minimizer completed in {end-start:.3f} seconds!!")
        d_opt = jnp.reshape(res.params, reshape)
        try:
            if d_ref.any():
                diff = jnp.linalg.norm(d_ref - d_opt)
        except AttributeError:
            pass
    return d_opt


logger = logging.getLogger(__name__)


def main():
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", True)

    global logger
    logging.basicConfig(filename="log.out", level=logging.ERROR, format="%(message)s")
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
        choices=["water", "acnit", "imidazole", "imidazole2", "imidazole3", "pyrazole"],
        help="Molecule type (with OpenMM files).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../benchmarks/OpenMM",
        help="Directory for benchmark input files.",
    )
    parser.add_argument(
        "--scf",
        type=str,
        default=None,
        help="SCF method, can be 'openmm' (for reference Dij) or None.",
    )

    args = parser.parse_args()
    dir = args.dir
    mol = args.mol
    scf = args.scf

    logger.info(f"%%%%%%%%%%% STARTING {mol.upper()} U_IND CALCULATION %%%%%%%%%%%%")
    logger.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

    Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = get_inputs(scf=scf, dir=dir, mol=mol, logger=logger)
    Dij = drudeOpt(
        Rij,
        jnp.ravel(Dij),
        Qi_shell,
        Qj_shell,
        Qi_core,
        Qj_core,
        u_scale,
        k,
        reshape=Dij.shape,
    )
    U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
    logger.info(f"OpenMM U_ind = {Uind_openmm:.7f} kJ/mol")
    logger.info(f"Python U_ind = {U_ind:.7f} kJ/mol")
    logger.info(f"{abs((Uind_openmm - U_ind) / U_ind) * 100:.2f}% Error")
    logger.info(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
    )


if __name__ == "__main__":
    main()
