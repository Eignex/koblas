package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngineApi

/**
 * Portable dense Level 2 and 3 under test. Core conformance never depends on an installed library.
 */
@OptIn(KoblasEngineApi::class)
internal val testDenseBlas: DenseBlas get() = BuiltinEngines.scalar

/**
 * Runs [body] against the mandatory portable seam, and against every other built-in composition this
 * platform has.
 *
 * The scheduling is shared, but the panel arithmetic inside its windows is not: a backend with a vector body
 * takes a different path through the same traversal, with its own tails and its own grouping. Running the
 * whole surface on each composition is what makes the conformance above cover those paths rather than only
 * the portable floor.
 */
@OptIn(KoblasEngineApi::class)
internal fun withDenseBlas(body: (DenseBlas) -> Unit) {
    body(BuiltinEngines.scalar)
    BuiltinEngines.simd?.let(body)
}
