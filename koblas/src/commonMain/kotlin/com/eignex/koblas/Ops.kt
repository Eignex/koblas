@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.sparse.SparseVectorKernels
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Visit each stored entry as (index, value), in ascending index order for any storage. A [SparseVector]
 * may present numerical zeros as stored, and any other [VectorLike] has every index visited.
 */
public inline fun VectorLike.forEachStored(block: (i: Int, v: Double) -> Unit) {
    when (this) {
        is DenseVector -> {
            val d = data
            for (i in 0 until d.size) block(i, d[i])
        }

        is SparseVector -> {
            val idx = indices
            val vals = values
            for (k in idx.indices) block(idx[k], vals[k])
        }

        else -> for (i in 0 until size) block(i, this[i])
    }
}

/** `aT * b`. Any sparse operand goes through [SparseVectorKernels], walking the stored entries only. */
public infix fun VectorLike.dot(other: VectorLike): Double {
    requireShape(size == other.size) { "size mismatch: $size vs ${other.size}" }
    if (this is DenseVector && other is DenseVector) {
        return koblas.vectorKernels.dot(data, 0, other.data, 0, size)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseVectorKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseVectorKernels.dot(this, other.data)
    if (this is DenseVector && other is SparseVector) return koblas.sparseVectorKernels.dot(other, data)
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun norm2(v: VectorLike): Double = when (v) {
    is DenseVector -> koblas.vectorKernels.nrm2(v.data, 0, v.size)
    is SparseVector -> koblas.sparseVectorKernels.nrm2(v)
    else -> euclideanNorm(v.toDoubleArray(), 0, v.size)
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun asum(v: VectorLike): Double = when (v) {
    is DenseVector -> koblas.vectorKernels.asum(v.data, 0, v.size)

    is SparseVector -> koblas.sparseVectorKernels.asum(v)

    else -> {
        var s = 0.0
        v.forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Index of the entry with maximal absolute value (BLAS `idamax`), -1 for a zero-length vector.
 * Ties resolve to the lowest index, and a vector with no stored entries returns 0.
 */
public fun iamax(v: VectorLike): Int {
    if (v.size == 0) return -1
    var best = -1
    var bestAbs = 0.0
    v.forEachStored { i, x ->
        val a = abs(x)
        if (a > bestAbs) {
            bestAbs = a
            best = i
        }
    }
    return if (best == -1) 0 else best
}

/** `dst = src` (BLAS `dcopy`). A sparse source zero-fills the destination first, so nothing survives. */
public fun copy(src: VectorLike, dst: DenseVector) {
    requireShape(src.size == dst.size) { "size mismatch: ${src.size} vs ${dst.size}" }
    when (src) {
        is DenseVector -> src.data.copyInto(dst.data)

        is SparseVector -> {
            dst.data.fill(0.0)
            koblas.sparseVectorKernels.scatter(src, dst.data)
        }

        else -> {
            dst.data.fill(0.0)
            src.forEachStored { i, v -> dst.data[i] = v }
        }
    }
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
public fun swap(a: DenseVector, b: DenseVector) {
    requireShape(a.size == b.size) { "size mismatch: ${a.size} vs ${b.size}" }
    val ad = a.data
    val bd = b.data
    for (i in ad.indices) {
        val t = ad[i]
        ad[i] = bd[i]
        bd[i] = t
    }
}

/**
 * @property c the cosine.
 * @property s the sine.
 * @property r the rotated length `±hypot(a, b)`, what `a` becomes as `b` goes to zero.
 */
public class Givens internal constructor(public val c: Double, public val s: Double, public val r: Double)

/**
 * Generate the plane rotation that zeroes [b] against [a] (BLAS `drotg`), rescaling so squares that would
 * overflow or vanish still rotate correctly. Netlib sign convention; the all-zero pair gives the identity.
 */
public fun rotg(a: Double, b: Double): Givens {
    if (b == 0.0 && a == 0.0) return Givens(c = 1.0, s = 0.0, r = 0.0)
    val absA = abs(a)
    val absB = abs(b)
    val scale = maxOf(absA, absB)
    val ra = a / scale
    val rb = b / scale
    val magnitude = scale * sqrt(ra * ra + rb * rb)
    val r = if (absA > absB) {
        if (a >= 0.0) magnitude else -magnitude
    } else {
        if (b >= 0.0) magnitude else -magnitude
    }
    return Givens(c = a / r, s = b / r, r = r)
}

/**
 * Apply a plane rotation (BLAS `drot`). Each pair `(x_i, y_i)` becomes `(c*x_i + s*y_i, c*y_i - s*x_i)`,
 * so both [x] and [y] are overwritten in place.
 */
public fun rot(x: DenseVector, y: DenseVector, rotation: Givens) {
    requireShape(x.size == y.size) { "size mismatch: ${x.size} vs ${y.size}" }
    val c = rotation.c
    val s = rotation.s
    if (c == 1.0 && s == 0.0) return
    val xd = x.data
    val yd = y.data
    for (i in xd.indices) {
        val xi = xd[i]
        val yi = yd[i]
        xd[i] = c * xi + s * yi
        yd[i] = c * yi - s * xi
    }
}

/** `y = y + alpha * x`. A sparse `x` touches only the positions it stores. */
public fun axpy(y: DenseVector, alpha: Double, x: VectorLike) {
    requireShape(y.size == x.size) { "size mismatch: ${y.size} vs ${x.size}" }
    if (alpha == 0.0) return
    when (x) {
        is DenseVector -> koblas.vectorKernels.axpy(y.data, 0, alpha, x.data, 0, y.size)
        is SparseVector -> koblas.sparseVectorKernels.axpy(y.data, alpha, x)
        else -> x.forEachStored { i, v -> y.data[i] += alpha * v }
    }
}

/** `v = alpha * v`. */
public fun scale(v: DenseVector, alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(v.data, 0, alpha, v.size)
}

/**
 * Rank-one update `A = A + alpha * x * yT` (BLAS `dger`), in place on [a]. Subtract by passing
 * `alpha = -1.0`.
 */
public fun ger(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix) {
    requireShape(a.rows == x.size && a.cols == y.size) {
        "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is DenseVector && y is DenseVector) {
        koblas.ger(alpha, x.data, y.data, a)
        return
    }
    val md = a.data
    val rows = a.rows
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val col = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> md[col + i] += scaled * xi }
        }
    }
}

/** Symmetric rank-1 update `A += alpha * x * xT` (BLAS `dsyr`), in place on [a]. See [Blas.syr]. */
public fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL): Unit = koblas.syr(
    alpha,
    x,
    a,
    uplo,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`), in place on [a]. See [Blas.syr2]. */
public fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL): Unit =
    koblas.syr2(alpha, x, y, a, uplo)

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1). This is the `anorm`
 * [LinearAlgebra.rcond] expects, computed before the matrix is factored.
 */
public fun norm1(a: DenseMatrix): Double {
    val ad = a.data
    val rows = a.rows
    var m = 0.0
    for (j in 0 until a.cols) {
        val base = j * rows
        var s = 0.0
        for (i in 0 until rows) s += abs(ad[base + i])
        if (s > m) m = s
    }
    return m
}

/** Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I). */
public fun normInf(a: DenseMatrix, workspace: Workspace? = null): Double {
    val rows = a.rows
    if (rows == 0 || a.cols == 0) return 0.0
    val sums = workspace?.take(rows) ?: DoubleArray(rows)
    sums.fill(0.0, 0, rows) // take() promises nothing about the contents
    val ad = a.data
    for (j in 0 until a.cols) {
        val base = j * rows
        for (i in 0 until rows) sums[i] += abs(ad[base + i])
    }
    var m = 0.0
    for (i in 0 until rows) if (sums[i] > m) m = sums[i]
    workspace?.release(sums)
    return m
}

/** Frobenius norm (LAPACK `dlange` with norm F). Rescales like [norm2] against overflow and underflow. */
public fun normFro(a: DenseMatrix): Double = euclideanNorm(a.data, 0, a.data.size)

/** Scale row i of [a] by d(i) in place, the product `D * A` for the diagonal D with entries [d]. */
public fun scaleRows(a: DenseMatrix, d: DoubleArray) {
    requireShape(d.size == a.rows) { "scaleRows: d length ${d.size} != ${a.rows} rows" }
    val rows = a.rows
    val ad = a.data
    for (j in 0 until a.cols) {
        val base = j * rows
        for (i in 0 until rows) ad[base + i] *= d[i]
    }
}

/** Scale column j of [a] by d(j) in place, the product `A * D` for the diagonal D with entries [d]. */
public fun scaleColumns(a: DenseMatrix, d: DoubleArray) {
    requireShape(d.size == a.cols) { "scaleColumns: d length ${d.size} != ${a.cols} columns" }
    val rows = a.rows
    for (j in 0 until a.cols) {
        val f = d[j]
        if (f != 1.0) koblas.vectorKernels.scale(a.data, j * rows, f, rows)
    }
}

/** Scale column j of [a] by d(j), in place, for a CSC matrix. The pattern is untouched. */
public fun scaleColumns(a: SparseMatrix, d: DoubleArray) {
    requireShape(d.size == a.cols) { "scaleColumns: d length ${d.size} != ${a.cols} columns" }
    for (j in 0 until a.cols) {
        val f = d[j]
        if (f == 1.0) continue
        for (k in a.colPtr[j] until a.colPtr[j + 1]) a.values[k] *= f
    }
}

/** Column [j] as a fresh vector, copied rather than viewed. */
public fun DenseMatrix.column(j: Int): DenseVector {
    requireIndex(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = j * rows
    return DenseVector.wrap(data.copyOfRange(start, start + rows))
}

/** Row [i] as a fresh vector, gathered across the backing. Prefer [column] where the algorithm allows. */
public fun DenseMatrix.row(i: Int): DenseVector {
    requireIndex(i in 0 until rows) { "row $i outside [0,$rows)" }
    val out = DoubleArray(cols)
    for (j in 0 until cols) out[j] = data[i + j * rows]
    return DenseVector.wrap(out)
}

/**
 * Fresh transposed matrix. For products, prefer the transpose flags on [LinearAlgebra.gemv] and
 * [LinearAlgebra.gemm], which read the original storage without copying.
 */
public fun DenseMatrix.transpose(): DenseMatrix {
    val t = DenseMatrix(cols, rows)
    val td = t.data
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) td[j + i * cols] = data[base + i]
    }
    return t
}

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly
 * stored zeros survive.
 */
public fun SparseMatrix.transpose(): SparseMatrix {
    val outPtr = IntArray(rows + 1)
    // Counts of row i land in outPtr(i + 1), then a prefix sum turns them into offsets.
    for (k in rowIdx.indices) outPtr[rowIdx[k] + 1]++
    for (i in 0 until rows) outPtr[i + 1] += outPtr[i]
    val outIdx = IntArray(values.size)
    val outVal = DoubleArray(values.size)
    // Walking source columns in order leaves each destination column's row indices ascending.
    val next = outPtr.copyOf()
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            val slot = next[rowIdx[k]]++
            outIdx[slot] = j
            outVal[slot] = values[k]
        }
    }
    return SparseMatrix(cols, rows, outPtr, outIdx, outVal)
}

/**
 * Matrix-vector product into a fresh dense result (BLAS `dgemv` with `alpha = 1`, `beta = 0`), for any
 * [MatrixLike] against any [VectorLike]. No transpose flag; [com.eignex.koblas.sparse.gemv] takes one.
 */
public fun MatrixLike.matVec(x: VectorLike): DenseVector {
    val a = this
    requireShape(a.cols == x.size) { "matVec shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    if (a is DenseMatrix && x is DenseVector) return DenseVector.wrap(koblas.gemv(a, x.data))
    val out = DenseVector(a.rows)
    val od = out.data
    if (a is DenseMatrix) {
        // One axpy per stored entry of x, down a contiguous column.
        val ad = a.data
        val rows = a.rows
        x.forEachStored { j, v ->
            if (v != 0.0) koblas.vectorKernels.axpy(od, 0, v, ad, j * rows, rows)
        }
    } else if (a is SparseMatrix) {
        // Accumulating column j of A scaled by x(j) reads only the stored entries of both operands.
        x.forEachStored { j, v ->
            if (v != 0.0) a.forEachInColumn(j) { i, aij -> od[i] += aij * v }
        }
    } else {
        // A foreign MatrixLike is read entry by entry.
        for (i in 0 until a.rows) {
            var s = 0.0
            x.forEachStored { j, v -> s += a[i, j] * v }
            od[i] = s
        }
    }
    return out
}
