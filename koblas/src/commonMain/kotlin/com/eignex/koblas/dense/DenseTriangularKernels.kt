@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

/**
 * The substitution over one diagonal block of a triangular matrix routine, across independent right-hand
 * sides.
 *
 * A triangular solve or multiply is a dependency chain down one index and nothing at all down the other. The
 * chain is the scheduling's: which block comes next, which direction the substitution runs in, and which
 * product removes a finished block from the ones still to come. What this backend owns is the arithmetic of
 * one diagonal block, where every right-hand side is doing the same substitution over the same coefficients
 * and none of them depends on another. That is the whole of what a vector unit has to work with here, and a
 * backend without one does the same arithmetic a value at a time.
 *
 * The block is `size` steps of the order by `sides` right-hand sides. Entry `(p, r)` of it is at
 * `xOffset + p · orderStride + r · rhsStride`, with `p` counted from the block's first step rather than from
 * the matrix's. Two strides rather than a leading dimension because a right-hand side is a column of the
 * block for a left-side call and a row of it for a right-side one, and a routine that only took the first
 * would have the caller transpose its data to say the second.
 *
 * The coefficients are `M(p, q) = T(start + q, start + p)` where the call transposed and
 * `T(start + p, start + q)` where it did not, read out of the caller's triangle where it lies. The `lower`
 * flag is `M`'s own triangle after that transposition and is what the substitution direction follows; it is
 * not the stored triangle of `T`, which a caller that transposed has already accounted for. A unit diagonal
 * is supplied rather than read, and no entry outside `M`'s selected triangle is read at all.
 *
 * Results agree with the direct definition to within rounding rather than bit for bit: an implementation may
 * fuse a multiply and an add. What it may not do is turn a division by the diagonal into a multiplication by
 * its reciprocal. Those differ by more than rounding at the extremes a solve is most often asked about: a
 * subnormal diagonal has no finite reciprocal to multiply by, and an infinite or zero one is a result the
 * caller is entitled to see rather than an error this contract detects.
 *
 * The block is written in place while the triangle is read, so the storage holding the right-hand sides must
 * be disjoint from the storage holding the triangle. These leaves check nothing; a public matrix call whose
 * triangle shares its block stages a copy of the triangle before it reaches one. Nothing else about the two
 * is assumed: the triangle may be any window of a larger array, and the sides may be strided through one.
 *
 * Implementations validate nothing and allocate nothing. A block with no steps or no right-hand sides does
 * nothing and reads nothing.
 */
public interface DenseTriangularKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /**
     * How many right-hand sides this backend recommends substituting at once, for a block of [order] steps
     * over [sides] of them.
     *
     * Always at least one and never more than [sides] when that is positive. It is a grouping and not a lane
     * count, and it is independent of the register tile, the panel grouping and the diagonal block the
     * scheduling chose: a diagonal block is as wide as the dependency chain is long, and this is how much of
     * the work beside it is done together.
     */
    public fun rightHandSideGroup(order: Int, sides: Int): Int

    /**
     * Whether a group of right-hand sides this far apart is worth copying into adjacent storage first.
     *
     * The copy is two passes over a block of the caller's data, out and back, and it buys a substitution
     * whose independent values lie next to each other. A backend whose arithmetic is a value at a time gains
     * nothing from that and answers false, and so does one whose right-hand sides are already adjacent.
     * Where this answers true the caller gathers, and the route says that a copy happened.
     */
    public fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean

    /**
     * Every arithmetic body a diagonal block of these extents reaches, in the order it reaches them.
     *
     * A list for the reason [DenseProductKernels.implementationsFor] is one: a block whose right-hand sides
     * do not fill a lane block may reach a different body for the remainder than for the rest.
     *
     * [contiguous] is part of the question rather than a detail of it: a vector body loads a lane block of
     * independent right-hand sides from consecutive entries, so a block whose right-hand sides are strided is
     * scalar work at any width unless the caller gathered it first.
     */
    public fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String>

    /**
     * Solves `M · X = X` in place over one diagonal block, for every right-hand side at once.
     *
     * Substitutes forward when [lower] and backward when it is not, dividing each finished step by its
     * diagonal coefficient unless [unitDiag]. The division is a division: a zero diagonal yields an infinity
     * or a NaN and a subnormal one yields whatever it yields, which is what the BLAS solves promise, since
     * they carry no status and report nothing.
     */
    public fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
    )

    /**
     * Multiplies `X = M · X` in place over one diagonal block, for every right-hand side at once.
     *
     * The order is the one that leaves no step needed after it has been overwritten, which runs opposite to
     * the solve's: a lower block is substituted from its last step upward, because a step is the sum of the
     * steps at or before it and those must still hold what they arrived with.
     */
    public fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
    )
}

/** `M(p, q)` for a diagonal block starting at [start], which is the entry a substitution reads. */
internal fun triangleEntry(t: DoubleArray, n: Int, start: Int, p: Int, q: Int, transposed: Boolean): Double =
    if (transposed) t[start + q + (start + p) * n] else t[start + p + (start + q) * n]
