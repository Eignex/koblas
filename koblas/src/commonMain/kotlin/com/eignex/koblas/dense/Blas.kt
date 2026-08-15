@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.VectorLike
import com.eignex.koblas.VectorView
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.transpose

/** Dense matrix routines as a backend half. */
public interface Blas : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val vectorKernels: VectorKernels get() = koblas.vectorKernels

    /** `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
     *  `beta == 0.0` overwrites [y] without reading it. */
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    )

    /** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *  `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /** `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     *  [Uplo.FULL] writes both triangles, unlike standard `dsyrk`; `beta == 0.0` overwrites without reading. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo = Uplo.FULL,
        workspace: Workspace? = null,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included; `beta == 0.0` overwrites [y] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    public fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true)

    /** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     *  [lower] triangle of [a] is read; `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    )

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`), the dense form a backend can dispatch. The free `ger` accepts
     *  [VectorView] operands and takes a sparse fast path. */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix)

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing the triangles [uplo] selects. [syrk] is the rank-k form. */
    public fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing the triangles [uplo] selects. */
    public fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL)

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     *  [transpose]. Writes the triangles [uplo] selects. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo = Uplo.FULL,
    )

    /** Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     *  `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out. */
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`). Flags follow [trsv];
     *  the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    )

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    public fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    )

    /** [gemv] over [DenseVector] operands, writing [y] in place. */
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean = false,
    ): Unit = gemv(alpha, a, x.data, beta, y.data, transpose)

    /** [gemv] over a [DenseVector], into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DenseVector, transpose: Boolean = false): DenseVector =
        DenseVector.wrap(gemv(a, x.data, transpose))

    /** [trsv] over a [DenseVector], solving in place. */
    public fun trsv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trsv(a, x.data, lower, transpose, unitDiag)

    /** [trmv] over a [DenseVector], multiplying [x] in place. */
    public fun trmv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trmv(a, x.data, lower, transpose, unitDiag)
}
