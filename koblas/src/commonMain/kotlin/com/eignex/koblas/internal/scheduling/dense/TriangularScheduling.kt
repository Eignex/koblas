@file:Suppress("LongParameterList") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.triangularProvider
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.triangle
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional
import com.eignex.koblas.internal.storage.dense.*

/*
 * Triangular solve and multiply from either side share one normalized triangle `M`.
 * A right-side call flips the transpose because `X · op(T) = B` is `op(T)ᵀ · x = b` per row.
 * Traversal and product windows then follow `M`'s dependencies.
 *
 * The diagonal block bounds substitution depth; the backend independently recommends a right-hand-side group.
 * Neither is a lane count, register tile or product cache block. Left-side right-hand sides are strided,
 * right-side ones adjacent. A backend-requested gathered copy is bounded by diagonal block times group,
 * regardless of the whole call's dimensions.
 */

/**
 * Diagonal substitution depth. Larger blocks move work from parallel products into the dependency chain
 * and increase the coefficient working set; this size belongs to the schedule.
 */
internal const val TRIANGULAR_DIAGONAL_BLOCK: Int = 64

/**
 * Substitute each diagonal block before its disjoint product update. Solves update remaining rows from
 * finished blocks; multiplies form the current block from still-untouched dependencies.
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
    // Solve in dependency order; multiply in reverse order to preserve input values until their last use.
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
        // The remaining dependency range receives a solve update or supplies the multiply inputs.
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
 * Triangular solve or multiply over coefficients already staged against overlap with [b].
 * Scale right-hand sides by [alpha] before substitution; other placements may round differently.
 */
internal fun triangularMatrix(
    engine: KoblasEngine,
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
    arrays: Scratch?,
    tOffset: Int = 0,
    ldt: Int = order,
    bOffset: Int = 0,
) {
    if (alpha != 1.0) {
        val rows = if (right) sides else order
        val cols = if (right) order else sides
        if (ldb == rows) {
            engine.kernels.scale(b, bOffset, alpha, rows * cols)
        } else {
            for (j in 0 until cols) engine.kernels.scale(b, bOffset + j * ldb, alpha, rows)
        }
    }
    val flip = transpose != right
    val mLower = lower != flip
    val orderStride = if (right) ldb else 1
    val rhsStride = if (right) 1 else ldb
    val block = if (TRIANGULAR_DIAGONAL_BLOCK < order) TRIANGULAR_DIAGONAL_BLOCK else order
    val triangles = engine.triangularProvider
    val group = triangularGroup(triangles, block, sides)
    val gathers = triangles.gathersRightHandSides(rhsStride, block, sides)
    arrays.borrowOptional(if (gathers) block * group else 0) { gathered ->
        forEachTriangularBlock(order, block, mLower, solve) {
                start,
                size,
                targetStart,
                targetCount,
                sourceStart,
                sourceCount,
            ->
            diagonalStep(
                triangles, t, ldt, tOffset, start, size, flip, mLower, unitDiag, solve,
                b, bOffset, orderStride, rhsStride, sides, group, gathers, gathered,
            )
            if (targetCount > 0 && sourceCount > 0) {
                triangularUpdate(
                    engine, if (solve) -1.0 else 1.0, t, ldt, tOffset, flip, b, bOffset, ldb,
                    sides, right, targetStart, targetCount, sourceStart, sourceCount, arrays,
                )
            }
        }
    }
}

/**
 * Cap the backend's recommended group by available sides, with a floor of one.
 * The recommendation controls residency for both gathered and natively adjacent right-hand sides.
 */
internal fun triangularGroup(triangles: EngineKernels, block: Int, sides: Int): Int {
    val recommended = triangles.rightHandSideGroup(block, sides)
    val bounded = if (recommended > sides) sides else recommended
    return if (bounded < 1) 1 else bounded
}

/**
 * Substitute one diagonal block in place or in gathered storage bounded by block order times side group,
 * independent of the whole call's dimensions.
 */
private fun diagonalStep(
    triangles: EngineKernels,
    t: DoubleArray,
    order: Int,
    tOffset: Int,
    start: Int,
    size: Int,
    transposed: Boolean,
    lower: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
    b: DoubleArray,
    bOffset: Int,
    orderStride: Int,
    rhsStride: Int,
    sides: Int,
    group: Int,
    gathers: Boolean,
    gathered: DoubleArray,
) {
    forEachPanel(sides, group) { first, lanes ->
        val origin = bOffset + start * orderStride + first * rhsStride
        if (!gathers) {
            substitute(
                triangles, solve, t, order, tOffset, start, size, transposed, lower, unitDiag,
                b, origin, orderStride, rhsStride, lanes,
            )
        } else {
            for (p in 0 until size) {
                val from = origin + p * orderStride
                val into = p * lanes
                for (lane in 0 until lanes) gathered[into + lane] = b[from + lane * rhsStride]
            }
            substitute(
                triangles, solve, t, order, tOffset, start, size, transposed, lower, unitDiag,
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
    triangles: EngineKernels,
    solve: Boolean,
    t: DoubleArray,
    order: Int,
    tOffset: Int,
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
            t, order, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
        )
    } else {
        triangles.diagonalMultiply(
            t, order, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
        )
    }
}

/**
 * Update through shared product scheduling: left-side calls use row strips with the triangle on the left;
 * right-side calls use column strips with it on the right. [triangleWindow] maps normalized `M` back to `T`.
 */
private fun triangularUpdate(
    engine: KoblasEngine,
    coefficient: Double,
    t: DoubleArray,
    order: Int,
    tOffset: Int,
    transposed: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    sides: Int,
    right: Boolean,
    targetStart: Int,
    targetCount: Int,
    sourceStart: Int,
    sourceCount: Int,
    arrays: Scratch?,
) {
    val triangle = DenseMatrix.wrap(
        if (transposed) sourceCount else targetCount,
        if (transposed) targetCount else sourceCount,
        t,
        tOffset + triangleWindow(order, transposed, targetStart, sourceStart),
        order,
    )
    if (right) {
        productWindow(
            engine, coefficient, DenseMatrix.wrap(sides, sourceCount, b, bOffset + sourceStart * ldb, ldb), false,
            triangle, !transposed, 1.0,
            DenseMatrix.wrap(sides, targetCount, b, bOffset + targetStart * ldb, ldb),
            OutputTriangle.Full, arrays,
        )
    } else {
        productWindow(
            engine, coefficient, triangle, transposed,
            DenseMatrix.wrap(sourceCount, sides, b, bOffset + sourceStart, ldb), false, 1.0,
            DenseMatrix.wrap(targetCount, sides, b, bOffset + targetStart, ldb),
            OutputTriangle.Full, arrays,
        )
    }
}

/**
 * Map a rectangle of normalized `M` to stored `T`, pairing its origin with the product's transpose flag.
 * A flipped orientation exchanges the rectangle's row and column starts.
 */
internal fun triangleWindow(order: Int, transposed: Boolean, rowStart: Int, columnStart: Int): Int =
    if (transposed) columnStart + rowStart * order else rowStart + columnStart * order

/**
 * The window each group of a selected-triangle traversal shares, beside the corner its own columns take.
 *
 * [fromDiagonal] says whether the last column's own diagonal entry belongs to that window: a rank update
 * writes it, while a symmetric product has already read it in the corner.
 */
internal inline fun forEachTrianglePanel(
    n: Int,
    group: Int,
    lower: Boolean,
    fromDiagonal: Boolean,
    action: (start: Int, width: Int, window: Int, rows: Int) -> Unit,
) {
    val diagonal = if (fromDiagonal) 1 else 0
    forEachPanel(n, group) { start, width ->
        val window = if (lower) start + width - diagonal else 0
        action(start, width, window, if (lower) n - window else start + diagonal)
    }
}

/**
 * Vector substitution depth, independent of [TRIANGULAR_DIAGONAL_BLOCK]. Inside each block is one dependency
 * chain; between blocks, a panel update can keep its destination in registers across the whole block.
 */
internal const val TRIANGULAR_VECTOR_BLOCK: Int = 64
