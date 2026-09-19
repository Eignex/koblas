package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngineApi

/**
 * Portable dense Level 2 and 3 under test. Core conformance never depends on an installed library.
 */
@OptIn(KoblasEngineApi::class)
internal val testDenseBlas: DenseBlas get() = BuiltinEngines.scalar

/**
 * Runs [body] against the mandatory portable seam.
 */
internal fun withDenseBlas(body: (DenseBlas) -> Unit): Unit = body(testDenseBlas)
