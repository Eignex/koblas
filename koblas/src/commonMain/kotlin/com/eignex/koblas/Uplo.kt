package com.eignex.koblas

/**
 * Output-triangle selector for [LinearAlgebra.syrk].
 *
 * [FULL] (the default) writes the complete, exactly symmetric result — the koblas extension over
 * BLAS. [LOWER] and [UPPER] follow `dsyrk` strictly: only the selected triangle (diagonal included)
 * is written and beta-scaled, and the opposite strict triangle is never read or touched.
 */
enum class Uplo { FULL, LOWER, UPPER }
