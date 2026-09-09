package com.eignex.koblas

import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.GeneralSparseLu
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
