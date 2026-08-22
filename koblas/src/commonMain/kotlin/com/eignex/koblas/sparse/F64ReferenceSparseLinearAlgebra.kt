package com.eignex.koblas.sparse

import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.factorization.ldl.*
import com.eignex.koblas.sparse.factorization.lu.*
import com.eignex.koblas.sparse.symbolic.*
import kotlin.math.abs

/**
 * The portable sparse backend, available on every target. The sparse seams declare their routines and this
 * implements them, so a binding that means to accelerate one cannot inherit the portable version by accident.
 */
@Suppress("TooManyFunctions") // the sparse surface a backend half covers
public object F64ReferenceSparseLinearAlgebra :
    F64SparseLinearAlgebra,
    F64SparseVectorKernels {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

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
        when {
            beta == 0.0 -> y.fill(0.0)
            beta != 1.0 -> for (i in y.indices) y[i] *= beta
        }
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

    override fun trsv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trsv")
        val n = a.rows
        requireShape(x.size == n) { "trsv: x length ${x.size} != $n" }
        // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        for (j in order) {
            if (!transpose) {
                val xj = if (unitDiag) x[j] else x[j] / diagonalOf(a, j)
                x[j] = xj
                if (xj != 0.0) {
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) x[i] -= v * xj
                    }
                }
            } else {
                var s = x[j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[i]
                }
                x[j] = if (unitDiag) s else s / diagonalOf(a, j)
            }
        }
    }

    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization =
        F64SparseLu.factorCsc(a, equilibrate, dropTolerance)

    override fun analyze(a: F64SparseMatrix, ordering: SparseOrdering): SparseSymbolic =
        SparseSymbolic.analyze(a, ordering)

    override fun ldl(a: F64SparseMatrix, policy: SparseLdlPolicy, ordering: SparseOrdering): F64SparseFactorization =
        numericLdl(a, analyze(a, ordering), policy)

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

    override fun nrm2(x: F64SparseVector): Double = euclideanNorm(x.values, 0, x.values.size)

    override fun asum(x: F64SparseVector): Double {
        var s = 0.0
        for (v in x.values) s += abs(v)
        return s
    }
}
