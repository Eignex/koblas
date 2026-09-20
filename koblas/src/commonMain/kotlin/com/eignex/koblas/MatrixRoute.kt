package com.eignex.koblas

import com.eignex.koblas.vendor.RouteKind

/**
 * What one matrix call executes, on the terms a dense and a sparse route share.
 *
 * Which of the two a product has is a fact about where its operands keep their values rather than about the
 * types a caller declared: the same generic product is dense scheduling between two dense operands and CSC
 * scheduling as soon as one of them is sparse. A query made with the operands therefore answers in the terms
 * both routes have, and a caller needing the rest of one reads the concrete route it was handed.
 */
public interface MatrixRoute {
    /** How the call was served. */
    public val kind: RouteKind

    /** The component that owns traversal, windows and arithmetic order. */
    public val scheduling: String

    /** The resolved entry point within [scheduling]. */
    public val entryPoint: String

    /** Every implementation and leaf the call reaches, in call order. */
    public val components: List<String>

    /** What the components cover, what the traversal keeps for itself, or why the call is composed. */
    public val reason: String?

    /**
     * Logical columns or right-hand sides the backend recommended handing over at a time, and zero where the
     * call groups none. A grouping, not a lane count.
     */
    public val executionGroup: Int

    /** The full attribution: the scheduling, plus every component it calls. */
    public val implementation: String

    /** Whether this call is admissible as an exact measurement of the implementations named. */
    public val exactlyMeasurable: Boolean
}
