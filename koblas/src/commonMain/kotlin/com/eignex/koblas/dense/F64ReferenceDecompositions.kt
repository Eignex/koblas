@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The portable dense factorizations, the semantic reference a native [F64Decompositions] is validated against.
 *
 * @param configured the kernels the inner loops use, or null to follow the [F64Context] default.
 */
internal class F64ReferenceDecompositions(private val configured: F64Kernels? = null) : F64Decompositions {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val kernels: F64Kernels get() = configured ?: koblas.kernels

    override fun ldl(a: F64DenseMatrix, workspace: Workspace?): F64LdlDecomposition =
        referenceLdl(kernels, a, workspace)

    override fun solveInto(ldl: F64LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        referenceLdlSolveInto(kernels, ldl, b, out)

    override fun qr(a: F64DenseMatrix, workspace: Workspace?): F64QrDecomposition = referenceQr(kernels, a)

    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition =
        referenceQrPivoted(kernels, a, tolerance)

    override fun applyQInto(qr: F64QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray =
        referenceApplyQInto(kernels, qr, y, out, transpose)

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return referenceLuFactorInto(kernels, a, F64LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        return referenceLuFactorInto(kernels, a, out)
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray = referenceLuSolveInto(kernels, lu, b, out, transpose, workspace)

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix = referenceLuSolveInto(kernels, lu, b, out, transpose, workspace)

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
    ): DoubleArray = referencePivotedLeastSquaresInto(kernels, qr, b, out, workspace)

    override fun solveLeastSquaresInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = referenceLeastSquaresInto(kernels, qr, b, out, workspace)

    override fun solveMinimumNormInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = referenceMinimumNormInto(kernels, qr, b, out, workspace)

    override fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val x = workspace?.take(n) ?: DoubleArray(n)
        val y = workspace?.take(n) ?: DoubleArray(n)
        val signs = workspace?.take(n) ?: DoubleArray(n)
        val probe = workspace?.take(n) ?: DoubleArray(n)
        try {
            return hagerEstimate(kernels, lu, anorm, n, x, y, signs, probe, workspace)
        } finally {
            workspace?.release(x)
            workspace?.release(y)
            workspace?.release(signs)
            workspace?.release(probe)
        }
    }

    override fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy): F64CholeskyDecomposition =
        referenceCholesky(kernels, a, policy)

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
                kernels,
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
        referenceSpdInvert(kernels, chol, workspace)
}
