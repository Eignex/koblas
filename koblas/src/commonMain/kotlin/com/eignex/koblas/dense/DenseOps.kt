@file:kotlin.jvm.JvmName("DenseBlas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.internal.portable.dense.*
import kotlin.jvm.JvmOverloads

/** `xᵀ · y`. */
public fun KoblasEngine.dot(x: DenseVector, y: DenseVector): Double {
    requireSameSize(x.size, y.size, "dot")
    return dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)
}

/** `‖x‖₂`, computed by the selected provider's overflow-avoiding method. */
public fun KoblasEngine.nrm2(x: DenseVector): Double = nrm2(x.values, x.offset, x.size, x.stride)

/** `Σ|xᵢ|`. */
public fun KoblasEngine.asum(x: DenseVector): Double = asum(x.values, x.offset, x.size, x.stride)

/**
 * The index of the entry of largest magnitude, or `-1` for an empty vector.
 *
 * BLAS defines its three reductions to return zero for a non-positive increment rather than to walk the
 * vector backwards, so a negatively strided vector is passed with the increment's magnitude and reaches
 * the same entries in the opposite order. For [nrm2] and [asum] the order does not matter. For this one it
 * decides ties: among entries of equal largest magnitude a forward vector reports the first and a backward
 * one reports the last. That is a real difference from a single-pass forward scan, and it is documented
 * rather than papered over because no vendor flag expresses the forward-scan choice.
 */
public fun KoblasEngine.iamax(x: DenseVector): Int = iamax(x.values, x.offset, x.size, x.stride)

/** `y = alpha · x + y`. */
public fun KoblasEngine.axpy(alpha: Double, x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "axpy")
    if (alpha == 0.0) return
    val source = x.stableFor(y)
    axpy(y.values, y.offset, alpha, source.values, source.offset, x.size, y.stride, source.stride)
}

/**
 * `x = alpha · x`.
 *
 * A negatively strided vector is passed with the increment's magnitude. BLAS defines `scal` to return
 * without doing anything for a non-positive increment, so keeping the sign would silently leave the vector
 * unscaled; scaling is order-independent, so walking the same entries forwards gives the same result.
 */
public fun KoblasEngine.scale(alpha: Double, x: DenseVector): Unit = scale(x.values, x.offset, alpha, x.size, x.stride)

/** `y = x`. */
public fun KoblasEngine.copy(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    if (sameVectorMapping(x, y)) return
    val contiguous = x.stride == 1 && y.stride == 1
    val provider = if (this is CombinedEngine) vectorProvider(EngineImplementation.Copy, x.size, contiguous) else this
    // Portable contiguous copy is a memmove, so it needs no alias snapshot.
    val source = if (provider === PortableEngine && contiguous) x else x.stableFor(y)
    provider.copy(source.values, source.offset, y.values, y.offset, x.size, source.stride, y.stride)
}

/** Exchanges the entries of [x] and [y]. */
public fun KoblasEngine.swap(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "swap")
    if (sameVectorMapping(x, y)) return
    withStablePair(x, y) { a, b -> swap(a.values, a.offset, b.values, b.offset, a.size, a.stride, b.stride) }
}

/** Applies [rot] through this engine, including its overlap and negative stride semantics. */
public fun KoblasEngine.rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
    requireSameSize(x.size, y.size, "rot")
    if (c == 1.0 && s == 0.0) return
    withStablePair(x, y) { a, b -> rot(a.values, a.offset, b.values, b.offset, a.size, c, s, a.stride, b.stride) }
}

/** Both inputs precede every write; at a shared entry the final write belongs to [y]. */
private inline fun withStablePair(x: DenseVector, y: DenseVector, operation: (DenseVector, DenseVector) -> Unit) {
    if (!x.overlaps(y)) return operation(x, y)
    val a = DenseVector.wrap(x.toDoubleArray())
    val b = DenseVector.wrap(y.toDoubleArray())
    operation(a, b)
    for (i in 0 until x.size) x[i] = a[i]
    for (i in 0 until y.size) y[i] = b[i]
}

/** [KoblasEngine.gemv] over whole arrays, which must match `op(A)` exactly. */
@JvmOverloads
public fun KoblasEngine.gemv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
    scratch: Scratch? = null,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    gemv(alpha, a, x, 0, 1, beta, y, 0, 1, transpose, scratch)
}

/** [KoblasEngine.gemv] over vectors, which must match `op(A)` exactly. */
@JvmOverloads
public fun KoblasEngine.gemv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    transpose: Boolean = false,
    scratch: Scratch? = null,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    gemv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, transpose, scratch)
}

/** [KoblasEngine.symv] over whole arrays of the matrix's order. */
@JvmOverloads
public fun KoblasEngine.symv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    requireSymvOperands(a, x.size, y.size)
    symv(alpha, a, x, 0, 1, beta, y, 0, 1, lower, scratch)
}

/** [KoblasEngine.symv] over vectors of the matrix's order. */
@JvmOverloads
public fun KoblasEngine.symv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    requireSymvOperands(a, x.size, y.size)
    symv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, lower, scratch)
}

/** [KoblasEngine.ger] over vectors spanning [a]'s rows and columns. */
@JvmOverloads
public fun KoblasEngine.ger(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrixRegion,
    scratch: Scratch? = null,
) {
    requireGerOperands(x.size, y.size, a)
    ger(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, scratch)
}

/** [KoblasEngine.ger] over whole arrays spanning [a]'s rows and columns. */
@JvmOverloads
public fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    scratch: Scratch? = null,
) {
    requireGerOperands(x.size, y.size, a)
    ger(alpha, x, 0, 1, y, 0, 1, a, scratch)
}

/**
 * `A(rowOffset + i, columnOffset + j) += alpha · x(xOffset + i) · y(yOffset + j)` for
 * `0 <= i < rows` and `0 <= j < columns`. Entries outside this rectangle are untouched.
 *
 * The destination retains [a]'s column-major leading dimension. Built-in panel kernels update it
 * directly, without copying the rectangle or allocating vector views. All matrix and vector ranges
 * are validated before writing, including when [alpha] is zero or either extent is empty.
 *
 * A vector sharing [a]'s buffer is snapshotted over only its selected range. A zero [alpha] or an empty
 * rectangle reads no entries and takes no scratch.
 */
@Suppress("LongParameterList") // the rank update and its three independently addressed windows
@JvmOverloads
public fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    xOffset: Int = 0,
    yOffset: Int = 0,
    scratch: Scratch? = null,
) {
    requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
    if (alpha == 0.0 || rows == 0 || columns == 0) return
    if (x === a.values || y === a.values) {
        val window = a.view(rowOffset, columnOffset, rows, columns)
        stagedWindows(scratch, x, xOffset, rows, 1, y, yOffset, columns, 1, window) {
                sx,
                sxo,
                sxs,
                sy,
                syo,
                sys,
            ->
            portableGer(alpha, sx, sxo, sxs, sy, syo, sys, window, window.offset, rows, columns)
        }
    } else {
        portableGer(
            alpha, x, xOffset, 1, y, yOffset, 1, a,
            a.offset + rowOffset + columnOffset * a.leadingDimension, rows, columns,
        )
    }
}

/** [KoblasEngine.syr] over a vector of the matrix's order. */
@JvmOverloads
public fun KoblasEngine.syr(
    alpha: Double,
    x: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    requireSyrOperands(a, x.size, "syr")
    syr(alpha, x.values, x.offset, x.stride, a, lower, scratch)
}

/** [KoblasEngine.syr2] over vectors of the matrix's order. */
@JvmOverloads
public fun KoblasEngine.syr2(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    requireSyr2Operands(a, x.size, y.size, "syr2")
    syr2(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, lower, scratch)
}

/** [KoblasEngine.trsv] over a vector of the triangle's order. */
@JvmOverloads
public fun KoblasEngine.trsv(
    a: DenseMatrixRegion,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    trsv(a, x.values, x.offset, x.stride, lower, transpose, unitDiag, scratch)
}

/** [KoblasEngine.trmv] over a vector of the triangle's order. */
@JvmOverloads
public fun KoblasEngine.trmv(
    a: DenseMatrixRegion,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    trmv(a, x.values, x.offset, x.stride, lower, transpose, unitDiag, scratch)
}
