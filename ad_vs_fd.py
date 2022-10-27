import argparse
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy
import scipy
from jax.config import config
from pyscf import gto, scf

import adscf

config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser(description='Comparison of AD and FD')
parser.add_argument('molecule', choices=['H2O', 'NH3', 'CH4'])
parser.add_argument('basis', choices=['STO-3G', '3-21G', 'cc-pVDZ'])
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

mol = gto.Mole()
mol.charge = 0
mol.spin = 0

# Structures are from https://www.molinstincts.com/
if args.molecule == 'H2O':
    mol.build(atom='H 0.0021 -0.0041 0.0020; O -0.0110 0.9628 0.0073; H 0.8669 1.3681 0.0011',
              basis=args.basis, unit='Angstrom')
elif args.molecule == 'NH3':
    mol.build(atom='N -0.0116 1.0048 0.0076; H 0.0021 -0.0041 0.0020; H 0.9253 1.3792 0.0006; H -0.5500 1.3634 -0.7668',
              basis=args.basis, unit='Angstrom')
elif args.molecule == 'CH4':
    mol.build(atom='H 0.0021 -0.0041 0.0020; C -0.0127 1.0858 0.0080; H 1.0099 1.4631 0.0003; H -0.5399 1.4469 -0.8751; H -0.5229 1.4373 0.9048',
              basis=args.basis, unit='Angstrom')

calcEnergy, gradEnergy = adscf.calcEnergy_create(mol)

start = time.time()
mf = scf.RHF(mol)
mf.scf()
elapsed_time = time.time() - start
print("SCF: {:.3f} ms".format(elapsed_time * 1000))
e_scf = scf.hf.energy_tot(mf)


# parameters
tau = 1.0
tau_m = 1e-10
tau_M = 1e10
rho = 1e-4
delta = 0.1
eta = 0.5
epsilon = 1e-3
max_iter = 10000

x_ad = []
y_ad = []
time_ad = []

start = time.time()
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
start_itr = time.time()
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
    x_ad.append(k)
    y_ad.append(calcEnergy(X)+mol.energy_nuc())
    time_ad.append(time.time() - start_itr)

    # Check loop condition (4)
    cond = jnp.linalg.norm(A @ X)
    if cond < epsilon:
        print("break at", k)
        break
    if k % 100 == 0:
        print("k:", k, "cond:", cond, "energy:", calcEnergy(X))

elapsed_time = time.time() - start
print("Automatic differentiation: {:.3f} ms".format(elapsed_time*1000))
e = calcEnergy(X)+mol.energy_nuc()
print(f"total energy = {e}")

x_fd = []
y_fd = []
time_fd = []

start = time.time()
# 1. initialize X0
S = mol.intor_symmetric('int1e_ovlp')  # overlap matrix
S64 = numpy.asarray(S, dtype=numpy.float64)
X_np = scipy.linalg.inv(scipy.linalg.sqrtm(S64))
X = jnp.asarray(X_np)

# 2. set C=f(X0) and Q0=1
C = calcEnergy(X)
Q = 1.0

# 3. calculate G0 and A0
G = scipy.optimize.approx_fprime(
    X.flatten(), calcEnergy, epsilon=1.49e-8).reshape((len(S), len(S)))
A = G @ X.T @ S - S @ X @ G.T

# function to calculate Y(tau)
I = jnp.identity(len(S))


def Y_tau(tau, X, A):
    return jnp.linalg.inv(I + 0.5 * tau * A @ S) @ (I - 0.5 * tau * A @ S) @ X


# main loop
start_itr = time.time()
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
    G_new = scipy.optimize.approx_fprime(
        X.flatten(), calcEnergy, epsilon=1.49e-8).reshape((len(S), len(S)))
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
    x_fd.append(k)
    y_fd.append(calcEnergy(X)+mol.energy_nuc())
    time_fd.append(time.time() - start_itr)

    # Check loop condition (4)
    cond = jnp.linalg.norm(A @ X)
    if cond < epsilon:
        print("break at", k)
        break
    if k % 100 == 0:
        print("k:", k, "cond:", cond, "energy:", calcEnergy(X))

elapsed_time = time.time() - start
print("Finite difference: {:.3f} ms".format(elapsed_time*1000))
e = calcEnergy(X)+mol.energy_nuc()
print(f"total energy = {e}")

if args.plot:
    p1 = plt.plot(x_ad, y_ad)
    p2 = plt.plot(x_fd, y_fd)
    plt.axhline(y=e_scf, linestyle='--', color='black')
    plt.xlabel("iteration", fontsize=16)
    plt.ylabel("total energy (Eh)", fontsize=16)
    plt.legend((p1[0], p2[0]),
               ("Automatic differentiation", "Finite difference"))
    plt.savefig(f"result-ad-vs-fd-{args.molecule}.png", dpi=300)
    plt.show()

    p1 = plt.plot(x_ad, time_ad)
    p2 = plt.plot(x_fd, time_fd)
    plt.xlabel("iteration", fontsize=16)
    plt.ylabel("Wall time (s)", fontsize=16)
    plt.legend((p1[0], p2[0]),
               ("Automatic differentiation", "Finite difference"))
    plt.savefig(f"result-ad-vs-fd-{args.molecule}-time.png", dpi=300)
    plt.show()
