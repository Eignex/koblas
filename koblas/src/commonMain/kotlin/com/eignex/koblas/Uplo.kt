package com.eignex.koblas

/**
 * Output-triangle selector for [LinearAlgebra.syrk].
 *
 * [FULL] (the default) writes the complete, exactly symmetric result — the koblas extension over
 * BLAS. [LOWER] and [UPPER] follow `dsyrk` strictly: only the selected triangle (diagonal included)
 * is written and beta-scaled, and the opposite strict triangle is never read or touched.
 */
enum class Uplo {
    /** Write the complete, exactly symmetric result; beta scales all of `C`. */
    FULL,

    /** Standard `dsyrk` with `uplo = L`: write and beta-scale the lower triangle only. */
    LOWER,

    /** Standard `dsyrk` with `uplo = U`: write and beta-scale the upper triangle only. */
    UPPER,
}
