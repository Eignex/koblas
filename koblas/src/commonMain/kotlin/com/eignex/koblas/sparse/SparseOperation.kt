package com.eignex.koblas.sparse

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
