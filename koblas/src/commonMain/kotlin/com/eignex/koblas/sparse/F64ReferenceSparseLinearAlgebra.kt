package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.axpyArithmetic
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.transposeCsc
import kotlin.math.abs

/**
 * A portable sparse backend, available on every target. The sparse seams declare their routines and this
 * implements them, so a binding that means to accelerate one cannot inherit the portable version by accident.
 *
 * @property configuredKernels dense vector kernels used by sparse matrix scaling, or null to follow [koblas].
 */
@Suppress("TooManyFunctions") // the sparse surface a backend half covers
public open class F64ReferenceSparseBackend(public val configuredKernels: F64Kernels? = null) :
    F64SparseLinearAlgebra,
    F64SparseKernels,
    F64GeneralSparseLu,
    F64SparseCholesky,
    F64SparseLdl,
    F64SparseQr,
    F64BasisFactorizations {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    private val denseKernels: F64Kernels get() = configuredKernels ?: koblas.kernels

    /** The product form over this backend's own factorization, so the portable half stays portable. */
    override fun basisSolver(a: F64SparseMatrix): F64BasisSolver = F64ProductFormBasisSolver(a, this)

    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        return F64RefactoringBasisFactorization(this, basis, factor(basis))
    }

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
        if (a.rows == 0 || a.cols == 0) return
        applyBeta(denseKernels, y, 0, y.size, beta)
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
                a.forEachInColumn(j) { i, v -> y[i] += v * xj }
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

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    override fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        right: Boolean,
    ) {
        val aRows = if (transposeA) a.cols else a.rows
        val aCols = if (transposeA) a.rows else a.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        val m = if (right) bRows else aRows
        val n = if (right) aCols else bCols
        requireShape(if (right) bCols == aRows else aCols == bRows) {
            val first = if (right) "${bRows}x$bCols" else "${aRows}x$aCols"
            val second = if (right) "${aRows}x$aCols" else "${bRows}x$bCols"
            "gemm: $first does not meet $second"
        }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        applyBeta(denseKernels, c.data, 0, c.data.size, beta)
        if (alpha == 0.0) return
        if (right) {
            multiplyFromTheRight(alpha, a, transposeA, b, transposeB, c, m)
        } else {
            multiplyFromTheLeft(alpha, a, transposeA, b, transposeB, c, m, n, aCols)
        }
    }

    /** `C += alpha · op(A) · op(B)`, which is [gemv] over the columns of the dense operand. */
    @Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
    private fun multiplyFromTheLeft(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        c: F64DenseMatrix,
        m: Int,
        n: Int,
        k: Int,
    ) {
        val cd = c.data
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
                    a.forEachInColumn(j) { i, v -> cd[cOff + i] += v * bjl }
                }
            }
        }
    }

    /**
     * `C += alpha · op(B) · op(A)`. Each stored entry of the sparse operand scales one column of the dense
     * one into one column of the destination, so the walk is over storage either way the sparse operand is
     * transposed and only which index names which column changes.
     */
    @Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
    private fun multiplyFromTheRight(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        c: F64DenseMatrix,
        m: Int,
    ) {
        val cd = c.data
        val bd = if (transposeB) {
            DoubleArray(b.data.size).also { transposeDense(b, it) }
        } else {
            b.data
        }
        val ld = if (transposeB) m else b.rows
        for (column in 0 until a.cols) {
            a.forEachInColumn(column) { row, v ->
                val cOff = (if (transposeA) row else column) * m
                val bColumn = if (transposeA) column else row
                axpyArithmetic(denseKernels, cd, cOff, alpha * v, bd, bColumn * ld, m)
            }
        }
    }

    override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix {
        requireShape(a.cols == b.rows) { "gemm: ${a.rows}x${a.cols} does not meet ${b.rows}x${b.cols}" }
        return multiplySparse(a, b)
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
        if (alpha != 1.0) denseKernels.scale(b.data, 0, alpha, b.data.size)
        if (right) {
            trsmRightCore(a, b, lower, !transpose, unitDiag)
        } else {
            for (l in 0 until b.cols) trsvCore(a, b.data, l * n, lower, transpose, unitDiag)
        }
    }

    /** Right solve over contiguous dense columns, preserving Netlib's zero-matrix coefficient guards. */
    private fun trsmRightCore(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val rows = b.rows
        if (rows == 0) return
        val bd = b.data
        val order = if (lower != transpose) 0 until a.rows else a.rows - 1 downTo 0
        for (j in order) {
            val jOff = j * rows
            if (!transpose) {
                if (!unitDiag) denseKernels.scale(bd, jOff, 1.0 / a[j, j], rows)
                a.forEachInColumn(j) { i, v ->
                    if (v != 0.0 && (if (lower) i > j else i < j)) {
                        denseKernels.axpy(bd, i * rows, -v, bd, jOff, rows)
                    }
                }
            } else {
                a.forEachInColumn(j) { i, v ->
                    if (v != 0.0 && (if (lower) i > j else i < j)) {
                        denseKernels.axpy(bd, jOff, -v, bd, i * rows, rows)
                    }
                }
                if (!unitDiag) denseKernels.scale(bd, jOff, 1.0 / a[j, j], rows)
            }
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
                val raw = x[offset + j]
                if (raw == 0.0) continue
                val xj = if (unitDiag) raw else raw / a[j, j]
                x[offset + j] = xj
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) x[offset + i] -= v * xj
                }
            } else {
                var s = x[offset + j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[offset + i]
                }
                x[offset + j] = if (unitDiag) s else s / a[j, j]
            }
        }
    }

    /** The portable factorization at its default policy; [F64ReferenceSparseDecompositions] carries the knobs. */
    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = F64ReferenceSparseDecompositions.factor(a)

    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization =
        F64ReferenceSparseDecompositions.cholesky(a)

    override fun ldl(a: F64SparseMatrix): F64SparseLdlFactorization = F64ReferenceSparseDecompositions.ldl(a)

    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization = F64ReferenceSparseDecompositions.qr(a)

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

/** Packs a dense transpose so sparse right products retain contiguous SIMD updates. */
private fun transposeDense(source: F64DenseMatrix, destination: DoubleArray) {
    for (j in 0 until source.cols) {
        for (i in 0 until source.rows) destination[j + i * source.cols] = source.data[i + j * source.rows]
    }
}

/** The shared portable sparse backend used by the process-wide registry. */
public object F64ReferenceSparseLinearAlgebra : F64ReferenceSparseBackend()
