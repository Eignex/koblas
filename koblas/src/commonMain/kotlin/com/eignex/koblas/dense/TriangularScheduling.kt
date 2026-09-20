@file:Suppress("LongParameterList") // a triangular matrix call carries its triangle, its block and four flags

package com.eignex.koblas.dense

import com.eignex.koblas.Workspace

/*
 * Shared scheduling for `B = alpha · op(T)⁻¹ · B` and `B = alpha · op(T) · B`, from either side.
 *
 * All eight flag combinations are one algorithm. Write the right-hand sides as `X(p, r)`, with `p` stepping
 * along the order of the triangle and `r` choosing one of the independent sides; then every variant is
 * `Σ M(p, q) · X(q, r)` over `M`'s own triangle, where `M(p, q)` is an entry of `T` read directly or
 * transposed. Which of the two it is depends on the side and the transpose together, because solving
 * `X · op(T) = B` for a row of `X` is solving `op(T)ᵀ · x = b` for a column, and the flags meet there.
 * Everything below that point — the direction of the substitution, which block updates which, and what the
 * product between them looks like — follows from `M`, so none of it is written out four times.
 *
 * The blocks are two and they are independent of each other and of everything under them. The diagonal block
 * is how much of the dependency chain one substitution covers, and it is this file's own number. The
 * right-hand sides handed over together are [DenseTriangularKernels.rightHandSideGroup]'s answer. Neither is
 * a lane count, a register tile or a cache block: the product between the blocks is an ordinary window of
 * shared product scheduling and picks its own geometry, and the substitution inside a diagonal block picks
 * its own grouping.
 *
 * The right-hand sides are left where the caller put them. A left-side call has them strided by the
 * destination's leading dimension and a right-side call has them adjacent, and the backend says whether
 * adjacency is worth a copy for the block it is about to substitute. Where it is, one diagonal block's worth
 * is gathered and written back, which is bounded by the two blocks above and not by the size of the call.
 */

/**
 * Steps of the order one diagonal substitution covers.
 *
 * The scheduling's own number. Below it is a dependency chain that nothing can overlap, so a larger block
 * means more of the call spent in a substitution and less of it in the products between them; above it the
 * substitution's own reads stop fitting anywhere useful. Sixty-four steps of a double-precision triangle is
 * thirty-two kilobytes of coefficients at the widest, which is the size at which the block being resident
 * still means something.
 */
internal const val TRIANGULAR_DIAGONAL_BLOCK: Int = 64

/**
 * The diagonal blocks and the product window after each, in the order the traversal reaches them.
 *
 * One traversal, walked by the execution below and by the route that describes it, for the reason
 * [forEachProductBlock] is one: a route that walked its own copy would name windows the call never cut.
 *
 * A solve finishes a block and then removes it from every step still to come, so its product target is the
 * rest of the order and its source is the block. A multiply forms a block from the steps it depends on,
 * which have not been touched yet, so its target is the block and its source is the rest. Either way the
 * diagonal block comes first and the product second, and the two windows are disjoint.
 */
internal inline fun forEachTriangularBlock(
    order: Int,
    block: Int,
    lower: Boolean,
    solve: Boolean,
    action: (
        start: Int,
        size: Int,
        targetStart: Int,
        targetCount: Int,
        sourceStart: Int,
        sourceCount: Int,
    ) -> Unit,
) {
    // A solve runs with the substitution: a lower system is finished from its first step. A multiply runs
    // against it, because a step of a lower product is the sum of the steps at or before it and those have
    // to still hold what they arrived with.
    val ascending = if (solve) lower else !lower
    var boundary = if (ascending) 0 else order
    while (if (ascending) boundary < order else boundary > 0) {
        val start = if (ascending) {
            boundary
        } else if (boundary > block) {
            boundary - block
        } else {
            0
        }
        val end = if (ascending) if (order - boundary > block) boundary + block else order else boundary
        val size = end - start
        // The part of the order this block still has business with is whatever the substitution has not
        // reached. For a solve that is the steps after it in dependency order, which it removes itself from;
        // for a multiply it is the same range, which it reads.
        val restStart = if (ascending) end else 0
        val restCount = if (ascending) order - end else start
        if (solve) {
            action(start, size, restStart, restCount, start, size)
        } else {
            action(start, size, start, size, restStart, restCount)
        }
        boundary = if (ascending) end else start
    }
}

/**
 * The groups of right-hand sides one diagonal block is substituted in, in the order it takes them.
 *
 * The backend's recommendation governs every call and not only the gathered ones. A copy is the reason the
 * grouping was worth asking about, but it is not the only thing it decides: the substitution reads its
 * block of sides once for every step before the one it is on, so how wide that block is decides what stays
 * resident whether the sides were copied or were already adjacent. Reporting one width while handing over
 * another would be the difference between a description and a guess, so there is one traversal here,
 * walked by the execution below and by the route that describes it.
 *
 * A backend that wants every side at once says so, which is what the portable one answers, and then this
 * is a single group. The last group of a call whose sides do not divide is shorter than the rest, and it is
 * the one a route that assumed they were all full would miss.
 */
internal inline fun forEachRightHandSideGroup(sides: Int, group: Int, action: (first: Int, lanes: Int) -> Unit) {
    var first = 0
    while (first < sides) {
        val lanes = if (group < sides - first) group else sides - first
        action(first, lanes)
        first += lanes
    }
}

/**
 * `B = alpha · op(T)⁻¹ · B` or `B = alpha · op(T) · B`, from either side, over a triangle already staged
 * against an overlap with [b].
 *
 * [alpha] is spent on the right-hand sides before the substitution rather than on the result after it, which
 * is what the reference definition does and is the same arithmetic: scaling a system's right-hand side
 * scales its solution, and scaling a product's operand scales the product.
 */
internal fun triangularMatrix(
    triangles: DenseTriangularKernels,
    products: DenseProductKernels,
    panels: DensePanelKernels,
    vectors: DenseVectorKernels,
    t: DoubleArray,
    order: Int,
    b: DoubleArray,
    ldb: Int,
    sides: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    workspace: Workspace?,
) {
    if (order == 0 || sides == 0) return
    if (alpha != 1.0) vectors.scale(b, 0, alpha, order * sides)
    val flip = transpose != right
    val mLower = lower != flip
    val orderStride = if (right) ldb else 1
    val rhsStride = if (right) 1 else ldb
    val block = if (TRIANGULAR_DIAGONAL_BLOCK < order) TRIANGULAR_DIAGONAL_BLOCK else order
    val group = triangularGroup(triangles, block, sides)
    val gathers = triangles.gathersRightHandSides(rhsStride, block, sides)
    workspace.borrowOptional(if (gathers) block * group else 0) { gathered ->
        forEachTriangularBlock(order, block, mLower, solve) {
                start,
                size,
                targetStart,
                targetCount,
                sourceStart,
                sourceCount,
            ->
            diagonalStep(
                triangles, t, order, start, size, flip, mLower, unitDiag, solve,
                b, orderStride, rhsStride, sides, group, gathers, gathered,
            )
            if (targetCount > 0 && sourceCount > 0) {
                triangularUpdate(
                    products, panels, if (solve) -1.0 else 1.0, t, order, flip,
                    b, ldb, sides, right, targetStart, targetCount, sourceStart, sourceCount, workspace,
                )
            }
        }
    }
}

/** Right-hand sides one diagonal substitution takes at a time, kept inside what the call actually has. */
internal fun triangularGroup(triangles: DenseTriangularKernels, block: Int, sides: Int): Int {
    val recommended = triangles.rightHandSideGroup(block, sides)
    val bounded = if (recommended > sides) sides else recommended
    return if (bounded < 1) 1 else bounded
}

/**
 * One diagonal block substituted, where the caller's right-hand sides lie or in a gathered copy of them.
 *
 * The copy is a block of the order by a group of the sides, so it is bounded by the two scheduling blocks
 * and not by the call: a solve over ten thousand right-hand sides gathers the same amount as one over a
 * hundred, as many times.
 */
private fun diagonalStep(
    triangles: DenseTriangularKernels,
    t: DoubleArray,
    order: Int,
    start: Int,
    size: Int,
    transposed: Boolean,
    lower: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
    b: DoubleArray,
    orderStride: Int,
    rhsStride: Int,
    sides: Int,
    group: Int,
    gathers: Boolean,
    gathered: DoubleArray,
) {
    forEachRightHandSideGroup(sides, group) { first, lanes ->
        val origin = start * orderStride + first * rhsStride
        if (!gathers) {
            substitute(
                triangles, solve, t, order, start, size, transposed, lower, unitDiag,
                b, origin, orderStride, rhsStride, lanes,
            )
        } else {
            for (p in 0 until size) {
                val from = origin + p * orderStride
                val into = p * lanes
                for (lane in 0 until lanes) gathered[into + lane] = b[from + lane * rhsStride]
            }
            substitute(
                triangles, solve, t, order, start, size, transposed, lower, unitDiag,
                gathered, 0, lanes, 1, lanes,
            )
            for (p in 0 until size) {
                val to = origin + p * orderStride
                val from = p * lanes
                for (lane in 0 until lanes) b[to + lane * rhsStride] = gathered[from + lane]
            }
        }
    }
}

/** Whichever of the two substitutions this call is, with the arguments they share. */
private fun substitute(
    triangles: DenseTriangularKernels,
    solve: Boolean,
    t: DoubleArray,
    order: Int,
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
) {
    if (solve) {
        triangles.diagonalSolve(
            t, order, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
        )
    } else {
        triangles.diagonalMultiply(
            t, order, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
        )
    }
}

/**
 * The product between a finished diagonal block and the steps it still has business with.
 *
 * An ordinary window of shared product scheduling, over both operands where they lie. From the left the
 * destination is a strip of rows of every right-hand side and the triangle is on the left of the product;
 * from the right it is a strip of columns and the triangle is on the right. The triangle's window is the
 * same rectangle of `M` either way, and which corner of `T` that rectangle is stored at is the one place the
 * transpose is read off.
 */
private fun triangularUpdate(
    products: DenseProductKernels,
    panels: DensePanelKernels,
    coefficient: Double,
    t: DoubleArray,
    order: Int,
    transposed: Boolean,
    b: DoubleArray,
    ldb: Int,
    sides: Int,
    right: Boolean,
    targetStart: Int,
    targetCount: Int,
    sourceStart: Int,
    sourceCount: Int,
    workspace: Workspace?,
) {
    val triangle = triangleWindow(order, transposed, targetStart, sourceStart)
    if (right) {
        productWindow(
            products, panels, coefficient,
            b, sourceStart * ldb, ldb, false,
            t, triangle, order, !transposed,
            1.0, b, targetStart * ldb, ldb,
            sides, targetCount, sourceCount, OutputTriangle.Full, workspace,
        )
    } else {
        productWindow(
            products, panels, coefficient,
            t, triangle, order, transposed,
            b, sourceStart, ldb, false,
            1.0, b, targetStart, ldb,
            targetCount, sides, sourceCount, OutputTriangle.Full, workspace,
        )
    }
}

/**
 * Where the rectangle of `M` with these row and column starts is stored in `T`.
 *
 * `M(i, q)` is `T(q, i)` where the call transposed and `T(i, q)` where it did not, so the rectangle's origin
 * is one corner or the other of the same stored square. The product reads it with a matching transpose flag,
 * which is why the offset and the flag are worked out together.
 */
internal fun triangleWindow(order: Int, transposed: Boolean, rowStart: Int, columnStart: Int): Int =
    if (transposed) columnStart + rowStart * order else rowStart + columnStart * order
