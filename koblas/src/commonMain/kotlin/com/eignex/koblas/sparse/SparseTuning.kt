package com.eignex.koblas.sparse

import com.eignex.koblas.internal.configuration.tunedInt

/**
 * Every dispatch crossover and search width the sparse routines apply, in one place and settable from
 * outside the build, as `DenseTuning` is for the dense half.
 *
 * Each resolves from a JVM system property, then an environment variable, then the compiled-in default. The
 * property is `koblas.sparse.<name>` and the variable is its upper-case spelling with dots replaced by
 * underscores. There are no system properties off the JVM, so a Native target reads the variable alone.
 * Every value is read once, when this object initializes.
 *
 * An override that does not parse, or that falls outside the range its entry allows, is ignored and the
 * default stands, for the reason given on `tunedIntValue`.
 *
 * Only retained sparse BLAS dispatch decisions belong here.
 */
internal object SparseTuning {
    /** Minimum support width for a JVM Vector API indexed load or store. */
    val simdIndexedCrossover: Int = tuned("indexed.simd.crossover", default = 16)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "sparse"
}
