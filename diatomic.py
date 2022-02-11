from pyscf import gto, scf, mp, cc, mcscf, ao2mo, fci
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.optimize import minimize
import numpy
import time

import jax.numpy as jnp
import jax.scipy.linalg
from jax import grad, jit, vmap, random

from jax.config import config
config.update("jax_enable_x64", True)

key = random.PRNGKey(0)

def calculate_charge_density(C, Ne):
    C_occ = C[:, :Ne//2]
    P = 2.0 * jnp.dot(C_occ, C_occ.T)
    return P

def calculate_FockMatrix(Hcore, JK, P):
    G = jnp.einsum('lk,ijkl->ij', P, JK) - 0.5 * jnp.einsum('lk,ilkj->ij', P, JK)
    return Hcore+G

def calculate_totalEnergy(P, Hcore, F):
    e1 = jnp.einsum('ji,ij->', P, Hcore)
    e2 = jnp.einsum('ji,ij->', P, F)
    return 0.5 * (e1+e2)

x = []
y = []

x_aug = []
y_aug = []

x_scf = []
y_scf = []

for i in range(2, 26):
    R = 0.1 * i
    print(f"interatomic distance: {R}")

    mol = gto.Mole()
    mol.charge = 0
    mol.spin = 0

    mol.build(atom = f'H 0.0 0.0 0.0; H 0.0 0.0 {R}', basis ='STO-3G', unit='Angstrom')

    # overlap matrix
    S = mol.intor_symmetric('int1e_ovlp')
    # kinetic energy
    T = mol.intor_symmetric('int1e_kin')
    # potential energy
    V = mol.intor_symmetric('int1e_nuc')
    Hcore = T + V
    # two electron integrals
    JK = ao2mo.restore(1, mol.intor('int2e'), mol.nao_nr())
    # the number of electrons
    Ne = mol.nelectron 

    @jit
    def calcEnergy(C):
        C = jnp.reshape(C, [len(S), len(S)])
        P = calculate_charge_density(C, Ne)
        F = calculate_FockMatrix(Hcore, JK, P)
        E = calculate_totalEnergy(P, Hcore, F)
        return E

    gradEnergy = jit(grad(calcEnergy))

    start = time.time()

    # PySCF calculation
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

    I = jnp.identity(len(S))

    # 1. initialize X0
    S64 = numpy.asarray(S, dtype=numpy.float64)
    X_np = scipy.linalg.inv(scipy.linalg.sqrtm(S64))
    X = jnp.asarray(X_np)
    
    # 2. set C=f(X0) and Q0=1
    C = calcEnergy(X)
    Q = 1.0

    # 4. calculate G0 and A0
    G = gradEnergy(X)
    A = G @ X.T @ S - S @ X @ G.T
    cond = jnp.linalg.norm(A @ X)

    # function to calculate Y(tau)
    def Y_tau(tau, X, A):
        return jnp.linalg.inv(I + 0.5 * tau * A @ S) @ (I - 0.5 * tau * A @ S) @ X
    
    # main loop
    for it in range(max_iter):
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
        G_new = gradEnergy(X_new)
        Sk = X_new - X
        Yk = G_new - G
        # 9
        if it % 2 == 0:
            tau_k = jnp.trace(Sk.T @ Sk) / abs(jnp.trace(Sk.T @ Yk))
        else:
            tau_k = abs(jnp.trace(Sk.T @ Yk)) / jnp.trace(Yk.T @ Yk)
        tau = max(min(tau_k, tau_M), tau_m)

        # Check loop condition
        X, Q, G = X_new, Q_new, G_new
        A = G @ X.T @ S - S @ X @ G.T
        cond = jnp.linalg.norm(A @ X)
        if cond < epsilon:
            break
    elapsed_time = time.time() - start
    print ("Curvilinear search: {:.3f} ms".format(elapsed_time*1000))
    e = calcEnergy(X)+mol.energy_nuc()
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
        res = minimize(jit(target), x0, jac=jit(grad(jit(target))), method="BFGS", options={'maxiter': 100})
        x0 = res.x
        constraint = orthogonality(x0)
        # 4
        lam += 2.0 * mu * constraint
        # 5
        mu *= 2.0
    elapsed_time = time.time() - start
    print ("Augmented: {:.3f} s".format(elapsed_time*1000))
    energy = res.fun+mol.energy_nuc()
    print(f"calculated energy = {energy}")
    x_aug.append(R)
    y_aug.append(energy)

p0 = plt.plot(x, y, marker="o")
p1 = plt.plot(x_aug, y_aug, marker="*")
p2 = plt.plot(x_scf, y_scf, marker="x")
plt.xlabel("interatomic distance (Å)", fontsize=16)
plt.ylabel("total energy (a.u.)", fontsize=16)
plt.legend((p0[0], p1[0], p2[0]), ("Curvilinear search", "Augmented Lagrangian", "PySCF"))
plt.savefig("result.png")
