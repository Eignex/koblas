package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.internal.portable.dense.TRIANGULAR_DIAGONAL_BLOCK

/**
 * Runs [body] against the mandatory portable seam, and against every other built-in composition this
 * platform has.
 *
 * The scheduling is shared, but the panel arithmetic inside its windows is not: a backend with a vector body
 * takes a different path through the same traversal, with its own tails and its own grouping. Running the
 * whole surface on each composition is what makes the conformance above cover those paths rather than only
 * the portable floor.
 */
internal fun withEngines(body: (KoblasEngine) -> Unit) {
    body(BuiltinEngines.scalar)
    BuiltinEngines.simd?.let(body)
}

/** Runs [body] against every composition of panel, product and substitution backends this platform has. */
internal fun withBackends(body: (EngineKernels, EngineKernels, EngineKernels) -> Unit) {
    body(PortableEngine, PortableEngine, PortableEngine)
    BuiltinEngines.simd?.let { body(it.panelProvider, it.productProvider, it.triangularProvider) }
}

/**
 * Right-hand sides enough to fill whatever the substitution backend groups them in, read from the backend
 * because a count that fills one machine's lane block leaves another's empty.
 */
internal val VECTOR_SIDES: Int = BuiltinEngines.simd
    ?.triangularProvider?.rightHandSideGroup(TRIANGULAR_DIAGONAL_BLOCK, UNGROUPED_SIDES)
    ?: 4

/** Wider than any grouping a backend recommends, so asking with it returns the recommendation itself. */
private const val UNGROUPED_SIDES = 1024
