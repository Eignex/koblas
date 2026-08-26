package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.forEachRow
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.internal.transposeCsc
import kotlin.math.abs

/**
 * The portable sparse backend, available on every target. The sparse seams declare their routines and this
 * implements them, so a binding that means to accelerate one cannot inherit the portable version by accident.
 */
@Suppress("TooManyFunctions") // the sparse surface a backend half covers
public object F64ReferenceSparseLinearAlgebra :
    F64SparseLinearAlgebra,
    F64SparseKernels {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** The product form over this backend's own factorization, so the portable half stays portable. */
    override fun basisSolver(a: F64SparseMatrix): F64BasisSolver = F64ProductFormBasisSolver(a, this)

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: F64SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        applyBeta(koblas.kernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                var s = 0.0
                a.forEachInColumn(j) { i, v -> s += v * x[i] }
                y[j] += alpha * s
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) a.forEachInColumn(j) { i, v -> y[i] += v * xj }
            }
        }
    }

    override fun transpose(a: F64SparseMatrix): F64SparseMatrix = transposeCsc(a)

    override fun trsv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trsv")
        val n = a.rows
        requireShape(x.size == n) { "trsv: x length ${x.size} != $n" }
        trsvCore(a, x, 0, lower, transpose, unitDiag)
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature
    override fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        applyBeta(koblas.kernels, cd, 0, cd.size, beta)
        if (alpha == 0.0) return
        val bd = b.data
        val ld = b.rows
        for (l in 0 until n) {
            val cOff = l * m
            if (transposeA) {
                for (i in 0 until m) {
                    var s = 0.0
                    a.forEachInColumn(i) { j, v -> s += v * bd[if (transposeB) l + j * ld else j + l * ld] }
                    cd[cOff + i] += alpha * s
                }
            } else {
                for (j in 0 until k) {
                    val bjl = alpha * bd[if (transposeB) l + j * ld else j + l * ld]
                    if (bjl != 0.0) a.forEachInColumn(j) { i, v -> cd[cOff + i] += v * bjl }
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        requireSquare(a, "trsm")
        val n = a.rows
        if (right) {
            requireShape(b.cols == n) { "trsm right: B has ${b.cols} cols, expected $n" }
        } else {
            requireShape(b.rows == n) { "trsm: B has ${b.rows} rows, expected $n" }
        }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (alpha != 1.0) koblas.kernels.scale(b.data, 0, alpha, b.data.size)
        if (right) {
            // Solving B·op(T)⁻¹ from the right is solving op(T)ᵀ against each row of B, so the same core runs
            // with the transpose flipped. The rows are gathered into scratch because a strided walk of a
            // column-major row would touch a cache line per entry.
            forEachRow(n, b) { row -> trsvCore(a, row, 0, lower, !transpose, unitDiag) }
        } else {
            for (l in 0 until b.cols) trsvCore(a, b.data, l * n, lower, transpose, unitDiag)
        }
    }

    /** [trsv] over the `n` entries of [x] from [offset], the columns of a right-hand side matrix being those. */
    @Suppress("LongParameterList") // the three BLAS triangle flags plus the segment
    private fun trsvCore(
        a: F64SparseMatrix,
        x: DoubleArray,
        offset: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val n = a.rows
        // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        for (j in order) {
            if (!transpose) {
                val xj = if (unitDiag) x[offset + j] else x[offset + j] / diagonalOf(a, j)
                x[offset + j] = xj
                if (xj != 0.0) {
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) x[offset + i] -= v * xj
                    }
                }
            } else {
                var s = x[offset + j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[offset + i]
                }
                x[offset + j] = if (unitDiag) s else s / diagonalOf(a, j)
            }
        }
    }

    /** The portable factorization at its default policy; [F64ReferenceSparseDecompositions] carries the knobs. */
    override fun factor(a: F64SparseMatrix): F64SparseFactorization = F64ReferenceSparseDecompositions.factor(a)

    override fun cholesky(a: F64SparseMatrix): F64SparseFactorization = F64ReferenceSparseDecompositions.cholesky(a)

    override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) s += vals[k] * y[idx[k]]
        return s
    }

    override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        var a = 0
        var b = 0
        while (a < x.indices.size && b < y.indices.size) {
            val ia = x.indices[a]
            val ib = y.indices[b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    s += x.values[a] * y.values[b]
                    a++
                    b++
                }
            }
        }
        return s
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) y[idx[k]] += alpha * vals[k]
    }

    override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) out[idx[k]] = vals[k]
    }

    override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) vals[k] = from[idx[k]]
    }

    override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) {
            val i = idx[k]
            vals[k] = from[i]
            from[i] = 0.0
        }
    }

    override fun nrm2(x: F64SparseVector): Double = euclideanNorm(x.values, 0, x.values.size)

    override fun asum(x: F64SparseVector): Double {
        var s = 0.0
        for (v in x.values) s += abs(v)
        return s
    }
}
