import jax.numpy as jnp
from jax import grad, jit
from pyscf import ao2mo


def calculate_charge_density(C, Ne):
    C_occ = C[:, :Ne//2]
    P = 2.0 * jnp.dot(C_occ, C_occ.T)
    return P


def calculate_FockMatrix(Hcore, JK, P):
    G = jnp.einsum('lk,ijkl->ij', P, JK) - 0.5 * \
        jnp.einsum('lk,ilkj->ij', P, JK)
    return Hcore+G


def calculate_totalEnergy(P, Hcore, F):
    e1 = jnp.einsum('ji,ij->', P, Hcore)
    e2 = jnp.einsum('ji,ij->', P, F)
    return 0.5 * (e1+e2)


def calcEnergy_create(mol):
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

    def calcEnergy_pre(C):
        C = jnp.reshape(C, [len(S), len(S)])
        P = calculate_charge_density(C, Ne)
        F = calculate_FockMatrix(Hcore, JK, P)
        E = calculate_totalEnergy(P, Hcore, F)
        return E
    calcEnergy = jit(calcEnergy_pre)
    gradEnergy = jit(grad(calcEnergy))
    return calcEnergy, gradEnergy
