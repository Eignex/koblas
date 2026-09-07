package com.eignex.koblas.sparse

import com.eignex.koblas.internal.backend.tunedDouble
import com.eignex.koblas.internal.backend.tunedInt

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
 * What is deliberately not here is the arithmetic. The sparse LU's pivot threshold and its drop tolerance
 * decide which entries are kept and how stable the factor is, so moving them changes the answer rather than
 * the time it takes to reach it. Those stay compiled in, where a review can see them, and only the
 * decisions that trade time against time are configurable.
 */
internal object SparseTuning {
    /**
     * Stored entries from which the dot against a dense vector repays crossing into the bundled C library.
     *
     * Measured twice on `SparseLevel1Benchmark`. At 163 stored entries the two are level, at 245 the C
     * leads by point estimate, and at 409 it leads beyond both error bars and holds 1.2x to 1.4x from
     * there to six thousand. This takes the length where the estimates turn.
     */
    val dotDenseCCrossover: Int = tuned("dot.dense.c.crossover", default = 256)

    /**
     * Candidate-bearing columns the Markowitz search examines before settling for the best pivot found,
     * which is the Suhl and Suhl bound.
     *
     * A wider search finds a sparser pivot and costs more to run, so this trades factorization time against
     * the fill that every later solve pays for. It changes which pivot is chosen and so the factor itself,
     * but not the accuracy of the solve, which the pivot threshold guards separately.
     */
    val luMaxCandidateColumns: Int = tuned("lu.max.candidate.columns", default = 4)

    /**
     * Density of the right-hand side above which a basis solve sweeps densely instead of tracking which
     * positions it reaches.
     *
     * Tracking costs a depth-first pass over the factor's column graph, which only pays while the reachable
     * set stays well under the order of the basis. A fraction, so the range is the unit interval and an
     * override outside it is ignored.
     */
    val reachableFtranMaxDensity: Double =
        tunedDouble(PREFIX, "reachable.ftran.max.density", default = 0.1, minimum = 0.0, maximum = 1.0)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "sparse"
}
