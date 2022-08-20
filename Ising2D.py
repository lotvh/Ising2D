#!/usr/bin/env python3

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from scipy.optimize import minimize
import jax
import matplotlib.pyplot as plt


def simple_time_evolution(ham, psi0):
    su_object = qtn.SimpleUpdate(
        psi0,
        ham,
        chi=32,
        compute_energy_every=10,
        compute_energy_per_site=True,
        keep_best=True,
    )

    for tau in [0.3, 0.1, 0.03, 0.01]:
        su_object.evolve(100, tau=tau)

    return su_object.best["energy"]


def autodiff_loss(psi, terms):
    psi.balance_bonds_()
    psi.equalize_norms_(1.0)

    return psi.compute_local_expectation(
        terms, max_bond=32, cutoff=0.0, normalized=True
    ) / (Lx * Ly)


def autodiff_opt(ham, psi0):
    tnopt = qtn.TNOptimizer(
        psi0,
        loss_fn=autodiff_loss,
        loss_constants={"terms": ham.terms},
        autodiff_backend="jax",
        optimizer="L-BFGS-B",
    )

    psi_opt = tnopt.optimize(100)

    return psi_opt.compute_local_expectation(
        ham.terms, normalized=True, max_bond=100
    ) / (Lx * Ly)


def plot_energy_bx(bx_range, res_energy, filename):
    plt.plot(bx_range, res_energy)
    plt.title("Ground State Energy of 2D-TFIM as a Function of Ext Field")
    plt.ylabel("Energy")
    plt.ylim(min(res_energy), max(res_energy))
    plt.xlabel("Strength of Ext Field")
    plt.xlim(min(bx_range), max(bx_range))
    plt.savefig(filename)


if __name__ == "__main__":

    # define system size
    Lx = 4
    Ly = 4

    opt_energy = []
    te_energy = []
    bx_range = np.linspace(0, 1, 11)
    for bx in bx_range:
        # define the Hamiltonian as the 2D-TFIM Hamiltonian:
        # H_\mathrm{Ising} = J \sum_{<ij>} \sigma^Z_i \sigma^Z_{j} - B_x \sum_{i} \sigma^X_i
        # for nearest neighbors <ij>
        j = 1.0
        ham = qtn.ham_2d_ising(Lx, Ly, j, bx)

        # initialize PEPS
        psi0 = qtn.PEPS.rand(Lx, Ly, bond_dim=4, seed=666)

        # run time evolution
        te = simple_time_evolution(ham, psi0)
        te_energy.append(te)

        # run optimizer
        opt_energy.append(autodiff_opt(ham, psi0))

    # plot the calculated energy as a function of the external field
    plot_energy_bx(bx_range, te_energy, "energy_vs_field_time_ev.pdf")
    plot_energy_bx(bx_range, opt_energy, "energy_vs_field_optimized.pdf")
