package com.eignex.koblas.sparse

import com.eignex.koblas.vendor.RouteKind

/**
 * A sparse Level 1 entry point, named so a caller can ask where a call of a given width would go.
 *
 * The width is the only route-relevant argument these operations have: every one of them walks a stored
 * support, and the choice between a vector kernel and its scalar fallback is made on how long that support is.
 */
public enum class SparseOperation(internal val entryPoint: String) {
    /** A sparse vector against a dense one, over an index and a value window. */
    DotDense("dotDense"),

    /** Two sparse vectors, merged in one pass over both index lists. */
    DotSparse("dotSparse"),

    /** A scaled sparse vector added into a dense destination. */
    Axpy("axpy"),

    /** Stored values written into a dense destination at their own positions. */
    Scatter("scatter"),

    /** Dense positions read back into a stored support. */
    Gather("gather"),

    /** [Gather] that also zeroes the positions it read. */
    GatherZero("gatherZero"),

    /** The Euclidean norm of a dense array at the positions an index window names. */
    IndexedNrm2("nrm2"),

    /** The Euclidean norm of a sparse vector's stored values. */
    Nrm2("nrm2"),

    /** The sum of magnitudes of a sparse vector's stored values. */
    Asum("asum"),
}

/**
 * What a concrete sparse Level 1 call does, derived from the decision the call itself makes.
 *
 * This is the sparse counterpart of [com.eignex.koblas.vendor.CallRoute], and it exists for the same reason: an
 * engine name is not evidence of what ran. A SIMD engine answers a call below its vector crossover, and a call
 * whose operation it never vectorised, by handing the whole thing to the scalar kernels. Timing that as a SIMD
 * measurement reports the scalar loop twice under two labels.
 *
 * [implementation] is therefore the fact a comparison has to read: it names what executed, not what was asked.
 * A route is about one call at one width, and building one allocates, so a caller asks once and then times.
 *
 * A [RouteKind.Composed] route is the admission that no kernel can be named yet, which the whole-vector norm
 * needs: it tries a square sum and abandons it for a rescaling loop when that leaves the normal range, so the
 * values decide. Naming either kernel would be a guess, and establishing it after the fact would mean tracing
 * inside the region being timed.
 */
public class SparseRoute internal constructor(
    /** The operation requested. */
    public val operation: SparseOperation,
    /** How the call was served. */
    public val kind: RouteKind,
    /**
     * The implementation that ran, which is not always the one that was asked.
     *
     * For a [RouteKind.Composed] route this is the selection that will choose, because no leaf can be named.
     */
    public val implementation: String,
    /** The resolved leaf within [implementation]. */
    public val entryPoint: String,
    /** Delegation detail, or null when the call stayed in the implementation that was asked. */
    public val adapter: String?,
    /** Why the call left the requested implementation, or null when it did not. */
    public val reason: String?,
) {
    /**
     * Whether this call is admissible as an exact measurement of the selection that was asked for.
     *
     * The question a benchmark arm asks before timing. A delegated call is still a real measurement of whatever
     * it reached, but it is not a measurement of the selection that was requested, so an exact arm declines it
     * and says why rather than publishing another implementation's time under its own name.
     */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct

    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        append(' ')
        append(kind.name.lowercase())
        append(' ')
        append(implementation)
        append(' ')
        append(entryPoint)
        adapter?.let { append(" via ").append(it) }
        reason?.let { append(" (").append(it).append(')') }
    }
}
