@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.F64Context
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.koblas
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape

/**
 * The portable dense factorizations, the semantic reference a native [F64Lapack] is validated against.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [F64Context] default.
 */
internal class F64ReferenceLapack(private val kernels: F64VectorKernels? = null) : F64Lapack {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val vectorKernels: F64VectorKernels get() = kernels ?: koblas.vectorKernels

    override fun ldl(a: F64DenseMatrix, workspace: Workspace?): F64LdlDecomposition =
        referenceLdl(vectorKernels, a, workspace)

    override fun solveInto(ldl: F64LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        referenceLdlSolveInto(vectorKernels, ldl, b, out)

    override fun qr(a: F64DenseMatrix, workspace: Workspace?): F64QrDecomposition = referenceQr(vectorKernels, a)

    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition =
        referenceQrPivoted(vectorKernels, a, tolerance)

    override fun applyQInto(qr: F64QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray =
        referenceApplyQInto(vectorKernels, qr, y, out, transpose)

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return referenceLuFactorInto(vectorKernels, a, F64LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        return referenceLuFactorInto(vectorKernels, a, out)
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray = referenceLuSolveInto(vectorKernels, lu, b, out, transpose, workspace)

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix = referenceLuSolveInto(vectorKernels, lu, b, out, transpose, workspace)

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    override fun solveInto(
        ldl: F64LdlDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(ldl.failedAt, "solve")
        val n = ldl.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        return solveColumnwise(b, out, n, nrhs, workspace) { col, dst -> solveInto(ldl, col, dst) }
    }

    override fun solveLeastSquaresInto(
        qr: F64PivotedQrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = referencePivotedLeastSquaresInto(vectorKernels, qr, b, out, workspace)

    override fun solveLeastSquaresInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = referenceLeastSquaresInto(vectorKernels, qr, b, out, workspace)

    override fun solveMinimumNormInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = referenceMinimumNormInto(vectorKernels, qr, b, out, workspace)

    override fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val x = workspace?.take(n) ?: DoubleArray(n)
        val y = workspace?.take(n) ?: DoubleArray(n)
        val signs = workspace?.take(n) ?: DoubleArray(n)
        val probe = workspace?.take(n) ?: DoubleArray(n)
        try {
            return hagerEstimate(vectorKernels, lu, anorm, n, x, y, signs, probe, workspace)
        } finally {
            workspace?.release(x)
            workspace?.release(y)
            workspace?.release(signs)
            workspace?.release(probe)
        }
    }

    override fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy): F64CholeskyDecomposition =
        referenceCholesky(vectorKernels, a, policy)

    /** Invert a general matrix from its LU factorization, returning `A⁻¹` given `P·A = L·U` (LAPACK `dgetri`).
     *  Prefer [solve] to apply `A⁻¹`, which costs less and is more accurate.
     *  @throws com.eignex.koblas.SingularMatrix if [lu] is singular; the position is [F64LuDecomposition.failedAt].
     */
    override fun invert(lu: F64LuDecomposition, workspace: Workspace?): F64DenseMatrix {
        if (lu.singular) {
            throw SingularMatrix(
                lu.failedAt,
                "invert: factorization is singular at pivot ${lu.failedAt}, so the inverse does not exist",
            )
        }
        val n = lu.n
        val inv = F64DenseMatrix.diagonal(n)
        return solveInto(lu, inv, inv, transpose = false, workspace = workspace)
    }

    /** Invert a triangular matrix into a fresh result (LAPACK `dtrtri`), returning `T⁻¹` for the [lower] or
     *  upper triangle of the square [a], taking the diagonal as 1 when [unitDiag].
     *  @throws com.eignex.koblas.SingularMatrix naming the first zero diagonal position.
     */
    override fun trtri(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean): F64DenseMatrix {
        requireShape(a.rows == a.cols) { "trtri requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (!unitDiag) {
            for (i in 0 until n) {
                if (a[i, i] == 0.0) {
                    throw SingularMatrix(i, "trtri: triangle is singular, diagonal entry $i is zero")
                }
            }
        }
        val inv = F64DenseMatrix(n, n)
        val invd = inv.data
        val column = DoubleArray(n)
        for (j in 0 until n) {
            column.fill(0.0)
            column[j] = 1.0
            trsvCore(
                vectorKernels,
                a.data,
                n,
                column,
                lower = lower,
                transpose = false,
                unitDiag = unitDiag,
            )
            column.copyInto(invd, n * j)
        }
        return inv
    }

    override fun invert(chol: F64CholeskyDecomposition, workspace: Workspace?): F64DenseMatrix =
        referenceSpdInvert(vectorKernels, chol, workspace)
}
