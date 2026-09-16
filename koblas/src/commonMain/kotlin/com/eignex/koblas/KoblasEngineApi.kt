package com.eignex.koblas

/**
 * Marks the seam that names an engine other than the one the platform selected.
 *
 * Production code wants [koblas], which is that selection. The engines behind this marker exist so a
 * benchmark can time one Level 1 implementation against another on the same case, and so a conformance test
 * can compare a vectorised kernel against the portable one that defines its semantics. Neither is a choice a
 * caller improves an application by making: the vectorised kernels already fall back to the portable ones
 * below their lane width and for any strided run, so the two are not alternatives at a given size.
 */
@RequiresOptIn(
    message = "Selecting an engine is for benchmarks and conformance tests; production code uses koblas",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.PROPERTY)
public annotation class KoblasEngineApi
