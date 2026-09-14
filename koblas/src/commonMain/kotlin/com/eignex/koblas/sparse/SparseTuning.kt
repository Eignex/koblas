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
    /**
     * Stored entries from which the dot against a dense vector repays crossing into the bundled C library.
     *
     * Re-measured on `koblas-bench` (`spdot-raw`) on an x86-64 AVX2 host across dense vectors of 65536 and
     * 262144 elements: the C leaf's eight independent accumulators only pay off once the indexed load
     * pattern outgrows L1 and starts hiding memory latency behind that instruction-level parallelism.
     * Below that point the foreign call is pure overhead on top of identical scalar throughput, and from
     * 256 to 2048 stored entries C measured 7% to 39% *slower* than the Kotlin loop, not faster. C first
     * leads clearly at 4096 stored entries (1.3x to 1.5x, both sizes) and holds from there through 8192.
     * This takes the length where the win becomes consistent rather than the old length where the two
     * were first estimated level, which sat inside the still-losing range.
     */
    val dotDenseCCrossover: Int = tuned("dot.dense.c.crossover", default = 4096)

    /** Indexed updates shorter than this remain in Kotlin to avoid a foreign call. */
    val cIndexedMutationCrossover: Int = tuned("indexed.mutation.c.crossover", default = 4096)

    /** Indexed norms shorter than this remain in Kotlin to avoid a foreign call. */
    val cIndexedNormCrossover: Int = tuned("indexed.norm.c.crossover", default = 4096)

    /** Minimum support width for a JVM Vector API indexed load or store. */
    val simdIndexedCrossover: Int = tuned("indexed.simd.crossover", default = 16)

    /** Minimum support width for a Kotlin/Native call into an indexed C leaf. */
    val nativeIndexedCrossover: Int = tuned("indexed.native.crossover", default = 32)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "sparse"
}
