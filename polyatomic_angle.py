import time
import numpy

import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo
import scipy
from scipy.optimize import minimize

import jax.numpy as jnp
from jax import grad, jit, random

from jax.config import config
config.update("jax_enable_x64", True)

import adscf

key = random.PRNGKey(0)

x = []
y = []

x_aug = []
y_aug = []

x_scf = []
y_scf = []

for i in range(105, 120):
    R = i
    print(f"interatomic angle: {R}")

    mol = gto.Mole()
    mol.charge = 0
    mol.spin = 0

    #mol.build(atom = f'O; H 1 0.96; H 1 0.96 2 {R}', basis ='ccpvdz', unit='Angstrom')
    mol.build(atom = f'N; H 1 1.01; H 1 1.01 2 107; H 1 1.01 2 107 3 {R}', basis ='ccpvdz', unit='Angstrom')

    calcEnergy, gradEnergy = adscf.calcEnergy_create(mol)

    start = time.time()

    # RHF energy calculation by PySCF
    mf = scf.RHF(mol)
    mf.scf()
    elapsed_time = time.time() - start
    print ("SCF: {:.3f} ms".format(elapsed_time * 1000))
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
    epsilon = 1e-3
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
    print ("Curvilinear search: {:.3f} ms".format(elapsed_time*1000))
    e = calcEnergy(X)+mol.energy_nuc()
    print(f"total energy = {e}")
    x.append(R)
    y.append(e)


p0 = plt.plot(x, y, marker="o")
p2 = plt.plot(x_scf, y_scf, marker="x")
plt.xlabel("dihedral angle (deg)", fontsize=16)
plt.ylabel("total energy (Eh)", fontsize=16)
plt.legend((p0[0], p2[0]), ("Curvilinear search", "PySCF"))
plt.gca().yaxis.get_major_formatter().set_useOffset(False)
plt.tight_layout()
plt.savefig("result.png", dpi=300)
