package com.eignex.koblas

import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.BasisFactorizations
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.QuasiDefiniteLdl
import com.eignex.koblas.sparse.RepeatedSparseLu
import com.eignex.koblas.sparse.SparseCholesky
import com.eignex.koblas.sparse.SparseQr
import com.eignex.koblas.sparse.basis.BasisSolvers

/** A typed key for one optional or independently selected capability in an [KoblasContext]. */
public class Capability<T : Backend> internal constructor(
    internal val select: (KoblasContext) -> T?,
    internal val named: (String) -> T?,
)

/** Typed capability keys, avoiding provider-specific casts in solver code. */
public object Capabilities {
    /** General pivoting sparse LU. */
    public val generalSparseLu: Capability<GeneralSparseLu> = slotCapability(BackendSlot.GeneralSparseLu)

    /** Repeated-pattern sparse LU, absent when no selected provider supports reuse. */
    public val repeatedSparseLu: Capability<RepeatedSparseLu> = slotCapability(BackendSlot.RepeatedSparseLu)

    /** Sparse Cholesky. */
    public val sparseCholesky: Capability<SparseCholesky> = slotCapability(BackendSlot.SparseCholesky)

    /** Sparse quasi-definite, numerically unpivoted `L * D * L^T`. */
    public val quasiDefiniteLdl: Capability<QuasiDefiniteLdl> = slotCapability(BackendSlot.QuasiDefiniteLdl)

    /** Sparse QR for least-squares solves. */
    public val sparseQr: Capability<SparseQr> = slotCapability(BackendSlot.SparseQr)

    /** Simplex basis factorization with column replacement. */
    public val basisFactorizations: Capability<BasisFactorizations> =
        slotCapability(BackendSlot.BasisFactorizations)

    /** Stateful simplex basis solvers. */
    public val basisSolvers: Capability<BasisSolvers> = slotCapability(BackendSlot.BasisSolvers)
}

/**
 * The key for the half [slot] fills, read out of a context or looked up by name through the same slot. Both
 * lookups hand back a [Backend] the slot already accepted, so the cast to the half's interface only recovers
 * a type the seam kept for it. A capability whose half is optional resolves to null rather than to the
 * placeholder standing in for it.
 */
private inline fun <reified T : Backend> slotCapability(slot: BackendSlot): Capability<T> = Capability(
    select = { slot.from(it) as? T },
    named = { BackendRegistry.named(slot, it) as? T },
)

/** Returns the provider selected for [capability], or null when the capability is optional and absent. */
public fun <T : Backend> KoblasContext.capability(capability: Capability<T>): T? = capability.select(this)

/** Returns the registered provider named [name] for [capability], without a provider-specific cast. */
public fun <T : Backend> backendNamed(name: String, capability: Capability<T>): T? = capability.named(name)

internal object MissingRepeatedSparseLu : Backend {
    override val name: String get() = "unavailable"
    override val isPortable: Boolean get() = true
    override val isAvailable: Boolean get() = false
}
