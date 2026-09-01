package com.eignex.koblas

import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/** A typed key for one optional or independently selected capability in an [F64Context]. */
public class F64Capability<T : Backend> internal constructor(
    internal val select: (F64Context) -> T?,
    internal val named: (String) -> T?,
)

/** Typed capability keys, avoiding provider-specific casts in solver code. */
public object F64Capabilities {
    /** General pivoting sparse LU. */
    public val generalSparseLu: F64Capability<F64GeneralSparseLu> = F64Capability(
        select = { it.generalSparseLu },
        named = BackendRegistry::generalSparseLuNamed,
    )

    /** Repeated-pattern sparse LU, absent when no selected provider supports reuse. */
    public val repeatedSparseLu: F64Capability<F64RepeatedSparseLu> = F64Capability(
        select = { it.repeatedSparseLu },
        named = BackendRegistry::repeatedSparseLuNamed,
    )

    /** Sparse Cholesky. */
    public val sparseCholesky: F64Capability<F64SparseCholesky> = F64Capability(
        select = { it.sparseCholesky },
        named = BackendRegistry::sparseCholeskyNamed,
    )

    /** Sparse quasi-definite, numerically unpivoted `L * D * L^T`. */
    public val quasiDefiniteLdl: F64Capability<F64QuasiDefiniteLdl> = F64Capability(
        select = { it.quasiDefiniteLdl },
        named = BackendRegistry::quasiDefiniteLdlNamed,
    )

    /** Sparse QR for least-squares solves. */
    public val sparseQr: F64Capability<F64SparseQr> = F64Capability(
        select = { it.sparseQr },
        named = BackendRegistry::sparseQrNamed,
    )

    /** Simplex basis factorization with column replacement. */
    public val basisFactorizations: F64Capability<F64BasisFactorizations> = F64Capability(
        select = { it.basisFactorizations },
        named = BackendRegistry::basisFactorizationsNamed,
    )

    /** Stateful simplex basis solvers. */
    public val basisSolvers: F64Capability<F64BasisSolvers> = F64Capability(
        select = { it.basisSolvers },
        named = BackendRegistry::basisSolversNamed,
    )
}

/** Returns the provider selected for [capability], or null when the capability is optional and absent. */
public fun <T : Backend> F64Context.capability(capability: F64Capability<T>): T? = capability.select(this)

/** Returns the registered provider named [name] for [capability], without a provider-specific cast. */
public fun <T : Backend> backendNamed(name: String, capability: F64Capability<T>): T? = capability.named(name)

internal object MissingRepeatedSparseLu : Backend {
    override val name: String get() = "unavailable"
    override val isPortable: Boolean get() = true
    override val isAvailable: Boolean get() = false
}
