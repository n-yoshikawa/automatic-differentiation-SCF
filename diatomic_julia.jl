using PyCall

pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
numpy = pyimport("numpy")
jax = pyimport("jax")
jnp = jax.numpy

jax.config.update("jax_enable_x64", true)

pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
adscf = pyimport("adscf")


function cayley(mol)
    # overlap matrix
    S = mol.intor_symmetric("int1e_ovlp")
    # kinetic energy
    T = mol.intor_symmetric("int1e_kin")
    # potential energy
    V = mol.intor_symmetric("int1e_nuc")
    Hcore = T + V
    # two electron integrals
    JK = pyscf.ao2mo.restore(1, mol.intor("int2e"), mol.nao_nr())
    # the number of electrons
    Ne = mol.nelectron 

    calcEnergy, gradEnergy = adscf.calcEnergy_create(S, Hcore, JK, Ne)

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
    S64 = numpy.asarray(S, dtype=numpy.float64)
    X_np = scipy.linalg.inv(scipy.linalg.sqrtm(S64))
    X = jnp.asarray(X_np)

    # 2. set C=f(X0) and Q0=1
    C = calcEnergy(X)
    Q = 1.0

    # 4. calculate G0 and A0
    G = gradEnergy(X)
    A = jnp.matmul(jnp.matmul(G, X.T), S) - jnp.matmul(jnp.matmul(S, X), G.T)

    # main loop
    for it=1:40
        Y_tau = adscf.Y_tau_create(S)
        Y = Y_tau(tau, X, A)
        A_norm = jnp.linalg.norm(A, "fro")
        X_old, Q_old, G_old = X, Q, G

        # 5
        while numpy.greater(calcEnergy(Y), C - rho * tau * A_norm^2.0)
            tau *= delta    # 6
            Y = Y_tau(tau, X, A)
        end

        # 8
        X_new = Y
        Q_new = eta * Q + 1.0
        C = (eta * Q * C + calcEnergy(X_new)) / Q_new
        G_new = gradEnergy(X_new)
        Sk = X_new - X
        Yk = G_new - G
        # 9
        if it % 2 == 0
            tau_k = jnp.trace(jnp.matmul(Sk.T, Sk)) / abs(jnp.trace(jnp.matmul(Sk.T, Yk)))
        else
            tau_k = abs(jnp.trace(jnp.matmul(Sk.T, Yk))) / jnp.trace(jnp.matmul(Yk.T, Yk))
        end
        tau = max(min(tau_k, tau_M), tau_m)

        # Check loop condition
        X, Q, G = X_new, Q_new, G_new
        A = jnp.matmul(jnp.matmul(G, X.T), S) - jnp.matmul(jnp.matmul(S, X), G.T)
        c = jnp.linalg.norm(jnp.matmul(A, X))
        if numpy.less(c, epsilon)
            println(it)
            break
        end
    end
    e = calcEnergy(X)+mol.energy_nuc()
    return convert(Float64, e)
end

function main()
    mol = pyscf.gto.Mole()
    mol.build(atom = "H 0.0 0.0 0.0; H 0.0 0.0 1.4", basis="STO-3G", unit="Angstrom")
    e = @time cayley(mol)
    println("Energy: ", e)
end

main()
