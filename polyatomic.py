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

mol = gto.Mole()
mol.charge = 0
mol.spin = 0

# Structures are from https://www.molinstincts.com/
mol.build(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 1.4', basis='ccpvdz', unit='Angstrom')
#mol.build(atom = 'N -0.0116 1.0048 0.0076; H 0.0021 -0.0041 0.0020; H 0.9253 1.3792 0.0006; H -0.5500 1.3634 -0.7668', basis='STO-3G', unit='Angstrom')
#mol.build(atom = 'H 0.0021 -0.0041 0.0020; C -0.0127 1.0858 0.0080; H 1.0099 1.4631 0.0003; H -0.5399 1.4469 -0.8751; H -0.5229 1.4373 0.9048', basis='STO-3G', unit='Angstrom')
#mol.build(atom = 'H 0.0021 -0.0041 0.0020; O -0.0110 0.9628 0.0073; H 0.8669 1.3681 0.0011', basis ='3-21G', unit='Angstrom')

calcEnergy, gradEnergy = adscf.calcEnergy_create(mol)

start = time.time()
mf = scf.RHF(mol)
mf.scf()
elapsed_time = time.time() - start
print ("SCF: {:.3f} ms".format(elapsed_time * 1000))
e_scf = scf.hf.energy_tot(mf)

start = time.time()

# overlap matrix
S = mol.intor_symmetric('int1e_ovlp')
S64 = numpy.asarray(S, dtype=numpy.float64)
X_np = scipy.linalg.inv(scipy.linalg.sqrtm(S64))
X = jnp.asarray(X_np)

G = gradEnergy(X)
A = G @ X.T @ S - S @ X @ G.T
cond = jnp.linalg.norm(A @ X)

C = calcEnergy(X)
Q = 1.0
tau = 1.0
delta = 0.1
rho = 1e-4
eta = 0.5
max_iter = 5000

I = jnp.identity(len(S))

def Y_tau(tau, X, A):
    return jnp.linalg.inv(I + 0.5 * tau * A @ S) @ (I - 0.5 * tau * A @ S) @ X

for it in range(max_iter):
    Y = Y_tau(tau, X, A)
    A_norm = jnp.linalg.norm(A, "fro")
    X_old, Q_old, G_old = X, Q, G

    while calcEnergy(Y) > C - rho * tau * A_norm**2.0:
        tau *= delta
        Y = Y_tau(tau, X, A)

    X_new = Y
    Q_new = eta * Q + 1.0
    C = (eta * Q * C + calcEnergy(X_new)) / Q_new
    G_new = gradEnergy(X_new)
    Sk = X_new - X
    Yk = G_new - G
    if it % 2 == 0:
        tau_k = jnp.trace(Sk.T @ Sk) / abs(jnp.trace(Sk.T @ Yk))
    else:
        tau_k = abs(jnp.trace(Sk.T @ Yk)) / jnp.trace(Yk.T @ Yk)
    tau = max(min(tau_k, 1e10), 1e-10)
    X, Q, G = X_new, Q_new, G_new
    A = G @ X.T @ S - S @ X @ G.T
    cond = jnp.linalg.norm(A @ X)
    if cond < 1e-3:
        print(it)
        break
elapsed_time = time.time() - start
print ("Curvilinear search: {:.3f} ms".format(elapsed_time*1000))
e = calcEnergy(X)+mol.energy_nuc()
print(f"total energy = {e}")

@jit
def orthogonality(x):
    C = jnp.reshape(x, [len(S), len(S)])
    return jnp.linalg.norm(C.transpose()@S@C - jnp.identity(len(S)))

start = time.time()
x0 = random.uniform(key, (S.size,))

mu = 1.0
lam = 0.0

constraint = orthogonality(x0)

while constraint > 1e-6:
    def target(x):
        h = orthogonality(x)
        return calcEnergy(x) + mu * h ** 2.0 + lam * h

    res = minimize(jit(target), x0, jac=jit(grad(jit(target))), method="BFGS", options={'maxiter': 100})
    x0 = res.x
    constraint = orthogonality(x0)
    lam += 2.0 * mu * constraint
    mu *= 2.0
elapsed_time = time.time() - start
print ("Augmented: {:.3f} s".format(elapsed_time*1000))
energy = res.fun+mol.energy_nuc()
print(f"calculated energy = {energy}")
