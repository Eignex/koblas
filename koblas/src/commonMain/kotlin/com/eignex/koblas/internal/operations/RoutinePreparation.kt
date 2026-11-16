@file:Suppress("LongParameterList") // Independent windows and extents.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.scheduling.dense.scheduleGer
import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.dense.*

/*
 * Shared portable and vendor preparation settles empty and zero-multiplier calls before arithmetic,
 * and snapshots overlapping inputs. Nonoverlapping vector views retain their addressing until the kernel.
 */

/** Shared GEMV/SYMV preparation supplies stable primitive matrix and vector windows to either backend. */
internal inline fun prepareMatrixVector(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrix,
    transpose: Boolean,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean?,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    if (lower == null) {
        requireGemvOperands(a, transpose, x.size, y.size)
    } else {
        requireSymvOperands(a, x.size, y.size)
    }
    if (rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        applyBeta(vectors, y.values, y.offset, rows, beta, y.stride)
        return
    }
    stagedRegionAgainst(arrays, a, y, lower = lower) { av, ao, ald ->
        stagedWindow(arrays, x.values, x.offset, x.size, x.stride, x.overlaps(y)) { values, offset, stride ->
            body(av, ao, ald, values, offset, stride)
        }
    }
}

/** Rank updates have distinct operand shapes even though they share alias preparation. */
internal enum class RankUpdate(val routine: String) {
    General("ger"),
    Symmetric("syr"),
    SymmetricTwo("syr2"),
}

/** Shared GER/SYR/SYR2 preparation snapshots identical vector mappings once. */
internal inline fun prepareRankUpdate(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrix,
    operation: RankUpdate,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    when (operation) {
        RankUpdate.General -> requireGerOperands(x.size, y.size, a)
        RankUpdate.Symmetric -> requireSyrOperands(a, x.size, operation.routine)
        RankUpdate.SymmetricTwo -> requireSyr2Operands(a, x.size, y.size, operation.routine)
    }
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    stagedWindows(
        arrays,
        x.values, x.offset, x.size, x.stride, regionWindowOverlap(a, x.values, x.offset, x.size, x.stride),
        y.values, y.offset, y.size, y.stride, regionWindowOverlap(a, y.values, y.offset, y.size, y.stride),
        body,
    )
}

/**
 * `op(T) · x = b` or `x = op(T) · x` in place. The substitution overwrites `x` as it goes, so a triangle
 * sharing that storage is snapshotted first.
 */
internal inline fun prepareTriangularVector(
    a: DenseMatrix,
    x: DenseVector,
    lower: Boolean,
    unitDiag: Boolean,
    what: String,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int) -> Unit,
) {
    requireTriangularVectorOperands(a, x.size, what)
    if (a.rows == 0) return
    stagedRegionAgainst(arrays, a, x, lower = lower, unitDiag = unitDiag, block = body)
}

/** Products settle no-read multipliers before lending snapshots; identical regions share one loan. */
internal inline fun prepareProduct(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    depth: Int,
    selected: Boolean?,
    aTriangle: Boolean?,
    vectors: KoblasEngine?,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    // Return before borrowing operand-sized scratch for an empty destination.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, selected, vectors)
        return
    }
    stagedRegion(arrays, a, c, lower = aTriangle) { av, ao, ald ->
        if (sameRegion(a, b)) {
            body(av, ao, ald, av, ao, ald)
        } else {
            stagedRegion(arrays, b, c) { bv, bo, bld -> body(av, ao, ald, bv, bo, bld) }
        }
    }
}

/** Stage triangular coefficients before scaling aliased right-hand sides can change them. */
internal inline fun prepareTriangularMatrix(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    what: String,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int) -> Unit,
) {
    requireTriangularMatrixOperands(a, b, right, what)
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    val sides = if (right) b.rows else b.cols
    if (sides == 0 || a.rows == 0) return
    stagedRegion(arrays, a, b, lower = lower, unitDiag = unitDiag, block = body)
}

/** Whole and rectangular GER share panel selection and staging without allocating unaliased views. */
internal fun KoblasEngine.updateRank(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrix,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    scratch: Scratch?,
) {
    requireVectorWindow(x.size, xOffset, rows, xStride, "x")
    requireVectorWindow(y.size, yOffset, columns, yStride, "y")
    if (alpha == 0.0 || rows == 0 || columns == 0) return
    val backend = kernels
    val destinationOffset = a.offset + rowOffset + columnOffset * a.leadingDimension
    if (x !== a.values && y !== a.values) {
        backend.scheduleGer(
            alpha, x, xOffset, xStride, y, yOffset, yStride, a.values, destinationOffset, a.leadingDimension,
            rows, columns,
        )
        return
    }
    val xAliased = regionWindowOverlap(
        a.values, destinationOffset, rows, columns, a.leadingDimension, x, xOffset, rows, xStride,
    )
    val yAliased = regionWindowOverlap(
        a.values, destinationOffset, rows, columns, a.leadingDimension, y, yOffset, columns, yStride,
    )
    stagedWindows(
        scratch, x, xOffset, rows, xStride, xAliased, y, yOffset, columns, yStride, yAliased,
    ) { xv, xo, xs, yv, yo, ys ->
        backend.scheduleGer(
            alpha, xv, xo, xs, yv, yo, ys, a.values, destinationOffset, a.leadingDimension, rows, columns,
        )
    }
}

/** Scales selected entries, never reading a zero-scaled destination or visiting its padding. */
internal fun scaleRegion(a: DenseMatrix, beta: Double, lower: Boolean? = null, vectors: KoblasEngine? = null) {
    if (beta == 1.0 || a.rows == 0 || a.cols == 0) return
    val backend = if (lower == null && beta != 0.0) vectors?.kernels else null
    scaleRegion(a.values, a.offset, a.rows, a.cols, a.leadingDimension, beta, lower, backend)
}

/** Scales a primitive destination window after its descriptor was unwrapped at entry. */
internal fun scaleRegion(
    a: DoubleArray,
    offset: Int,
    rows: Int,
    columns: Int,
    lda: Int,
    beta: Double,
    lower: Boolean? = null,
    vectors: KernelProvider? = null,
) {
    if (beta == 1.0 || rows == 0 || columns == 0) return
    if (lower == null && vectors != null) {
        if (lda == rows) {
            applyBeta(vectors, a, offset, rows * columns, beta)
        } else {
            for (j in 0 until columns) applyBeta(vectors, a, offset + j * lda, rows, beta)
        }
        return
    }
    for (j in 0 until columns) {
        val first = selectedRowStart(j, rows, lower)
        val last = selectedRowEnd(j, rows, lower)
        for (i in first until last) {
            val at = offset + i + j * lda
            a[at] = if (beta == 0.0) 0.0 else beta * a[at]
        }
    }
}

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: KoblasEngine, v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int = 1) {
    applyBeta(v, off, len, beta, stride) { k.kernels.vectorKernels.scale(v, off, beta, len, stride) }
}

/** Primitive scaling preserves the selected backend without passing the public adapter into schedules. */
internal fun applyBeta(k: KernelProvider, v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int = 1) {
    applyBeta(v, off, len, beta, stride) { k.vectorKernels.scale(v, off, beta, len, stride) }
}

/** Resolve scaling only when needed: zero overwrites and identity leaves the window untouched. */
private inline fun applyBeta(v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int, scale: () -> Unit) {
    when {
        beta == 0.0 -> if (stride == 1) {
            v.fill(0.0, off, off + len)
        } else {
            for (i in 0 until len) v[off + i * stride] = 0.0
        }

        beta != 1.0 -> scale()
    }
}
