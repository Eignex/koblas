package com.eignex.koblas

import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64BasisFactorizations
import com.eignex.koblas.sparse.F64GeneralSparseLu
import com.eignex.koblas.sparse.F64QuasiDefiniteLdl
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.F64SparseCholesky
import com.eignex.koblas.sparse.F64SparseQr
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/** A typed key for one optional or independently selected capability in an [F64Context]. */
public class F64Capability<T : Backend> internal constructor(
    internal val select: (F64Context) -> T?,
    internal val named: (String) -> T?,
)

/** Typed capability keys, avoiding provider-specific casts in solver code. */
public object F64Capabilities {
    /** General pivoting sparse LU. */
    public val generalSparseLu: F64Capability<F64GeneralSparseLu> = slotCapability(BackendSlot.F64GeneralSparseLu)

    /** Repeated-pattern sparse LU, absent when no selected provider supports reuse. */
    public val repeatedSparseLu: F64Capability<F64RepeatedSparseLu> = slotCapability(BackendSlot.F64RepeatedSparseLu)

    /** Sparse Cholesky. */
    public val sparseCholesky: F64Capability<F64SparseCholesky> = slotCapability(BackendSlot.F64SparseCholesky)

    /** Sparse quasi-definite, numerically unpivoted `L * D * L^T`. */
    public val quasiDefiniteLdl: F64Capability<F64QuasiDefiniteLdl> = slotCapability(BackendSlot.F64QuasiDefiniteLdl)

    /** Sparse QR for least-squares solves. */
    public val sparseQr: F64Capability<F64SparseQr> = slotCapability(BackendSlot.F64SparseQr)

    /** Simplex basis factorization with column replacement. */
    public val basisFactorizations: F64Capability<F64BasisFactorizations> =
        slotCapability(BackendSlot.F64BasisFactorizations)

    /** Stateful simplex basis solvers. */
    public val basisSolvers: F64Capability<F64BasisSolvers> = slotCapability(BackendSlot.F64BasisSolvers)
}

/**
 * The key for the half [slot] fills, read out of a context or looked up by name through the same slot. Both
 * lookups hand back a [Backend] the slot already accepted, so the cast to the half's interface only recovers
 * a type the seam kept for it. A capability whose half is optional resolves to null rather than to the
 * placeholder standing in for it.
 */
private inline fun <reified T : Backend> slotCapability(slot: BackendSlot): F64Capability<T> = F64Capability(
    select = { slot.from(it) as? T },
    named = { BackendRegistry.named(slot, it) as? T },
)

/** Returns the provider selected for [capability], or null when the capability is optional and absent. */
public fun <T : Backend> F64Context.capability(capability: F64Capability<T>): T? = capability.select(this)

/** Returns the registered provider named [name] for [capability], without a provider-specific cast. */
public fun <T : Backend> backendNamed(name: String, capability: F64Capability<T>): T? = capability.named(name)

internal object MissingRepeatedSparseLu : Backend {
    override val name: String get() = "unavailable"
    override val isPortable: Boolean get() = true
    override val isAvailable: Boolean get() = false
}
