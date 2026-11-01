package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.configuration.tunedInt
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * Sparse schedule/workspace tuning and the standard indexed profile's crossover override.
 * Entries resolve once from a JVM system property, then an environment variable, then the default.
 * Invalid values retain the default; backend choice otherwise belongs to engine profiles.
 */
internal object SparseTuning {
    /** Minimum support width for a JVM Vector API indexed load or store. */
    val simdIndexedCrossover: Int = tuned("indexed.simd.crossover", default = 16)

    /**
     * The most right-hand sides the sparse scheduling will hold staged adjacent at once.
     *
     * A ceiling over the backend's recommendation rather than the width itself: how many right-hand sides a
     * panel wants is the backend's question, since it is the one with the body, and how much of a dense
     * block may be copied into a workspace buffer while one sparse column is walked is the scheduling's.
     * Where the recommendation is below this, this changes nothing.
     */
    val contiguousRhsPanel: Int = tuned("rhs.panel", default = 64)

    /**
     * Stored entries per dense element copied, below which a sparse product keeps the caller's layout.
     *
     * Staging a panel of right-hand sides into adjacent order costs a copy of the dense operand, or of the
     * destination twice over, and buys vector arithmetic over the entries the traversal walks. The ratio
     * between those two is this number, and a product too sparse to pay it runs where its operands already
     * are.
     *
     * The measured ratio is four: the band just below it was no better than the caller's own layout,
     * and above it the copy pays and keeps paying as the support thickens. A gathering product copies half
     * as much as a scattering one, so the same ratio admits it at half the density, which is what a ratio
     * rather than a density is for.
     */
    val stagedRhsCrossover: Int = tuned("rhs.staging.crossover", default = 4)

    /**
     * Rows of range per row touched, below which a discovered column is swept into order rather than sorted.
     *
     * A column of a sparse product collects its rows in the order the contributing columns held them and has
     * to emit them ascending. Sorting costs with what the column holds; sweeping the marks costs with the
     * range the column could have reached, whatever it found there. This is where the two cross.
     */
    val supportSweepFactor: Int = tuned("support.sweep.factor", default = 8, minimum = 0)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "sparse"
}
