import argparse
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy
import scipy
from jax import grad, jit, random
from jax.config import config
from pyscf import gto, scf
from scipy.optimize import minimize

import adscf

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(
    description='Draw potential energy curves for a diatomic molecule.')
parser.add_argument('molecule', choices=['H2', 'HF'])
parser.add_argument('basis', choices=['STO-3G', '3-21G'])
args = parser.parse_args()

key = random.PRNGKey(0)

x = []
y = []

x_aug = []
y_aug = []

x_scf = []
y_scf = []

min_range = 2 if args.molecule == 'H2' else 5
max_range = 26 if args.molecule == 'H2' else 31

for i in range(min_range, max_range):
    R = 0.1 * i
    print(f"interatomic distance: {R:.2f}")

    mol = gto.Mole()
    mol.charge = 0
    mol.spin = 0

    if args.molecule == 'H2':
        mol.build(atom=f'H 0.0 0.0 0.0; H 0.0 0.0 {R:.2f}',
                  basis=args.basis, unit='Angstrom')
    elif args.molecule == 'HF':
        mol.build(atom=f'H 0.0 0.0 0.0; F 0.0 0.0 {R:.2f}',
                  basis=args.basis, unit='Angstrom')

    calcEnergy, gradEnergy = adscf.calcEnergy_create(mol)

    start = time.time()

    # RHF energy calculation by PySCF
    mf = scf.RHF(mol)
    mf.scf()
    elapsed_time = time.time() - start
    print("SCF: {:.3f} ms".format(elapsed_time * 1000))
    e_scf = scf.hf.energy_tot(mf)
    x_scf.append(R)
    y_scf.append(e_scf)

    # Curvilinear search using Cayley transformation
    start = time.time()

    # parameters
    tau = 1.0
    tau_m = 1e-10
    tau_M = 1e10
    rho = 1e-4
    delta = 0.1
    eta = 0.5
    epsilon = 1e-6
    max_iter = 5000

    # 1. initialize X0
    S = mol.intor_symmetric('int1e_ovlp')  # overlap matrix
    S64 = numpy.asarray(S, dtype=numpy.float64)
    X_np = scipy.linalg.inv(scipy.linalg.sqrtm(S64))
    X = jnp.asarray(X_np)

    # 2. set C=f(X0) and Q0=1
    C = calcEnergy(X)
    Q = 1.0

    # 3. calculate G0 and A0
    G = gradEnergy(X)
    A = G @ X.T @ S - S @ X @ G.T

    # function to calculate Y(tau)
    I = jnp.identity(len(S))

    def Y_tau(tau, X, A):
        return jnp.linalg.inv(I + 0.5 * tau * A @ S) @ (I - 0.5 * tau * A @ S) @ X

    # main loop
    for k in range(max_iter):
        Y = Y_tau(tau, X, A)
        A_norm = jnp.linalg.norm(A, "fro")
        X_old, Q_old, G_old = X, Q, G

        # 5
        while calcEnergy(Y) > C - rho * tau * A_norm**2.0:
            tau *= delta    # 6
            Y = Y_tau(tau, X, A)

        # 8
        X_new = Y
        Q_new = eta * Q + 1.0
        C = (eta * Q * C + calcEnergy(X_new)) / Q_new

        # 9
        G_new = gradEnergy(X_new)
        A_new = G_new @ X_new.T @ S - S @ X_new @ G_new.T

        # 10
        Sk = X_new - X
        Yk = G_new - G
        if k % 2 == 0:
            tau_k = jnp.trace(Sk.T @ Sk) / abs(jnp.trace(Sk.T @ Yk))
        else:
            tau_k = abs(jnp.trace(Sk.T @ Yk)) / jnp.trace(Yk.T @ Yk)
        tau = max(min(tau_k, tau_M), tau_m)

        # Update variables for next iteration
        X, Q, G, A = X_new, Q_new, G_new, A_new

        # Check loop condition (4)
        cond = jnp.linalg.norm(A @ X)
        if cond < epsilon:
            break

    elapsed_time = time.time() - start
    print("Curvilinear search: {:.3f} ms".format(elapsed_time*1000))
    e = calcEnergy(X) + mol.energy_nuc()
    print(f"total energy = {e}")
    x.append(R)
    y.append(e)

    # augmented Lagrangian
    @jit
    def orthogonality(x):
        C = jnp.reshape(x, [len(S), len(S)])
        return jnp.linalg.norm(C.transpose()@S@C - jnp.identity(len(S)))

    start = time.time()
    x0 = random.uniform(key, (S.size,))

    # 1
    mu = 1.0
    lam = 0.0

    constraint = orthogonality(x0)

    # 2
    while constraint > 1e-6:
        def target(x):
            h = orthogonality(x)
            return calcEnergy(x) + mu * h ** 2.0 + lam * h

        # 3
        res = minimize(jit(target), x0, jac=jit(grad(jit(target))),
                       method="BFGS", options={'maxiter': 100})
        x0 = res.x
        constraint = orthogonality(x0)
        # 4
        lam += 2.0 * mu * constraint
        # 5
        mu *= 2.0
    elapsed_time = time.time() - start
    print("Augmented: {:.3f} s".format(elapsed_time*1000))
    energy = res.fun+mol.energy_nuc()
    print(f"calculated energy = {energy}\n")
    x_aug.append(R)
    y_aug.append(energy)

p2 = plt.plot(x_scf, y_scf, marker="o")
p1 = plt.plot(x_aug, y_aug, marker="*")
p0 = plt.plot(x, y, marker="x")
plt.xlabel("interatomic distance (Å)", fontsize=16)
plt.ylabel("total energy (Eh)", fontsize=16)
location = 'upper right' if args.molecule == 'H2' else 'lower right'
plt.legend((p0[0], p1[0], p2[0]),
           ("Curvilinear search", "Augmented Lagrangian", "PySCF"),
           loc=location)
plt.savefig(f"result-{args.molecule}-{args.basis}.png", dpi=300)
plt.show()
