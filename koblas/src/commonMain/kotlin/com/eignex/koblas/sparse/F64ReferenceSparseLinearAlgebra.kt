package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.internal.multiplyFromTheLeft
import com.eignex.koblas.sparse.internal.multiplyFromTheRight
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.stableFor
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.trmmLeftCore
import com.eignex.koblas.sparse.internal.trmmRightCore
import com.eignex.koblas.sparse.internal.trmvCore
import com.eignex.koblas.sparse.internal.trsmLeftCore
import com.eignex.koblas.sparse.internal.trsmRightCore
import com.eignex.koblas.sparse.internal.trsvCore
import com.eignex.koblas.sparse.internal.withExplicitDiagonal
import kotlin.math.abs

/** Dense right-hand sides processed per walk of portable CSC storage. */
internal const val REFERENCE_SPARSE_RHS_WIDTH: Int = 4

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
    F64QuasiDefiniteLdl,
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
        requireGemvShape(a, transpose, x.size, y.size)
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
        trsvCore(a, x, lower, transpose, unitDiag)
    }

    override fun trmv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trmv")
        val n = a.rows
        requireShape(x.size == n) { "trmv: x length ${x.size} != $n" }
        trmvCore(a.stableFor(x), x, lower, transpose, unitDiag)
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
        workspace: Workspace?,
    ) {
        // Multiplying the dense operand by the sparse one from the right is this product with the operands
        // the other way round, so the same derivation answers both.
        val (m, k, n) = if (right) {
            requireGemmShape(b, transposeB, a, transposeA, c)
        } else {
            requireGemmShape(a, transposeA, b, transposeB, c)
        }
        applyBeta(denseKernels, c.data, 0, c.data.size, beta)
        if (alpha == 0.0) return
        if (right) {
            multiplyFromTheRight(denseKernels, alpha, a, transposeA, b, transposeB, c, m, workspace)
        } else {
            multiplyFromTheLeft(alpha, a, transposeA, b, transposeB, c, m, n, k, workspace)
        }
    }

    /** `C += alpha · op(A) · op(B)`, reusing each walk of the sparse operand over a small RHS panel. */

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
        workspace: Workspace?,
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
        if (n == 0) return
        if (alpha != 1.0) denseKernels.scale(b.data, 0, alpha, b.data.size)
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        if (right) {
            withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                trsmRightCore(denseKernels, a, b, lower, !transpose, diagonal)
            }
        } else {
            workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
                withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                    trsmLeftCore(a, b, lower, transpose, diagonal, work)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        requireSquare(a, "trmm")
        val n = a.rows
        if (right) {
            requireShape(b.cols == n) { "trmm right: B has ${b.cols} cols, expected $n" }
        } else {
            requireShape(b.rows == n) { "trmm: B has ${b.rows} rows, expected $n" }
        }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        val triangle = a.stableFor(b.data)
        if (alpha != 1.0) denseKernels.scale(b.data, 0, alpha, b.data.size)
        // Read once for every right-hand side rather than once per trmvCore call, as trsm does.
        val diagonal = if (unitDiag) null else DoubleArray(n) { triangle[it, it] }
        if (right) {
            trmmRightCore(denseKernels, triangle, b, lower, transpose, unitDiag, diagonal)
        } else {
            trmmLeftCore(triangle, b, lower, transpose, unitDiag, diagonal)
        }
    }

    /** The portable factorization at its default policy; [F64ReferenceSparseDecompositions] carries the knobs. */
    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = F64ReferenceSparseDecompositions.factor(a)

    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization =
        F64ReferenceSparseDecompositions.cholesky(a)

    override fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        F64ReferenceSparseDecompositions.quasiDefiniteLdl(a)

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

/** The shared portable sparse backend used by the process-wide registry. */
public object F64ReferenceSparseLinearAlgebra : F64ReferenceSparseBackend()
