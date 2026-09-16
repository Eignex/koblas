@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * Dense matrix routines, served by the vendor BLAS the platform selected.
 *
 * Arithmetic is the library's, not Koblas's. Koblas validates shapes and refuses operand overlap the standard
 * leaves undefined, then makes one whole call; what the library does with a zero multiplier, an infinity, a
 * subnormal, or the order it accumulates in is the library's contract, and it can differ between vendors as it
 * differs between builds of one vendor. Code that needs a stronger guarantee than BLAS gives has to own it.
 *
 * These are Koblas's own shapes for the same routines: array operands, booleans for the transpose and the
 * stored triangle, and overloads that allocate a result. [com.eignex.koblas.vendor.Blas] is the standard's own
 * shape underneath, and the transpose and structure travel beside an operand as flags there rather than being
 * inferred from it.
 *
 * On the JVM every operand is copied into native memory for the downcall, so a measurement of one of these
 * calls includes that copy. Kotlin/Native pins the caller's storage and passes it in place.
 *
 * Every operation here but [transpose], which is a storage transform the standard has no entry point for,
 * needs a library. On a host without one they raise
 * [com.eignex.koblas.vendor.MissingVendorException]; containers, Level 1 and the sparse primitives do not.
 */
public interface DenseBlas {
    /** `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose]. */
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

    /**
     * Fresh transposed [a]. For a product prefer the transpose flags on [gemv] and [gemm], which avoid a
     * caller-visible intermediate; this is for a caller that means to hold the transpose.
     *
     * A storage transform rather than arithmetic, so it stays a Kotlin loop: the standard has no entry point
     * for it, and the BLAS-like `omatcopy` extension is not one every supported library exports.
     */
    public fun transpose(a: DenseMatrix): DenseMatrix

    /** `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *  [c] must not share a buffer with either input. */
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

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed
     * as `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read
     * nor written. [c] must not share a buffer with either input.
     *
     * Not every library exports it. Where one does not, the call is composed from `gemm` plus a triangle copy,
     * and the route of the call reports which of the two ran rather than leaving the name to imply the first.
     */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    public fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     * Only the [lower] or upper triangle is written. [c] must not share a buffer with [a].
     */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    public fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
    )

    /** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     *  [lower] triangle of [a] is read; [c] must not share a buffer with either input. */
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

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`). */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix)

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. [x] must be dense or
     *  strided storage, which is what a vendor can address. */
    public fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean = true)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. Operand
     *  storage follows [syr]. */
    public fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean = true)

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Writes only the [lower] or upper triangle. [c] must not share a buffer with either input.
     */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
    )

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     * `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out.
     *
     * The diagonal is divided by, not tested: `dtrsv` carries no `info` and reports nothing, so a singular
     * triangle yields infinities or NaNs and the caller who needs the distinction tests the diagonal first.
     */
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags
     *  follow [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    )

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    public fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of
     *  [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    )
}
