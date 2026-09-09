@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*

/** Dense matrix routines bound to one immutable kernel implementation. */
public interface Blas {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /** The vector kernels this implementation's shared routines run on. */
    public val kernels: Kernels

    /** `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
     *  `beta == 0.0` overwrites [y] without reading it. */
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    )

    /** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /**
     * [gemv] over borrowed strided storage. The destination must not overlap [a] or [x]; disjoint views may
     * share one backing buffer. Implementations may pass offsets, strides, and leading dimensions directly
     * to BLAS and must not materialize a contiguous copy. Like BLAS `dgemv`, a zero row or column count
     * returns without reading or changing [y].
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    public fun gemv(
        alpha: Double,
        a: StridedMatrixView,
        x: StridedVectorView,
        beta: Double,
        y: StridedVectorView,
        transpose: Boolean = false,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        require(!y.overlaps(x) && !a.overlaps(y)) { "gemv: destination overlaps an input view" }
        if (a.rows == 0 || a.cols == 0) return
        for (i in 0 until y.size) {
            y[i] = when (beta) {
                0.0 -> 0.0
                1.0 -> y[i]
                else -> beta * y[i]
            }
        }
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                var sum = 0.0
                for (i in 0 until a.rows) sum += a[i, j] * x[i]
                y[j] += alpha * sum
            }
        } else {
            for (j in 0 until a.cols) {
                // Formed even when the multiplier is zero, so an infinite coefficient still yields NaN the
                // way the dense reference and netlib do. Skipping the column would hide it.
                val multiplier = alpha * x[j]
                for (i in 0 until a.rows) y[i] += multiplier * a[i, j]
            }
        }
    }

    /** [gemv] over borrowed storage into a fresh owned array. */
    public fun gemv(a: StridedMatrixView, x: StridedVectorView, transpose: Boolean = false): DoubleArray {
        val result = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, StridedVectorView(result, 0, result.size), transpose)
        return result
    }

    /**
     * Fresh transposed [a]. For a product prefer the transpose flags on [gemv] and [gemm], which avoid a
     * caller-visible intermediate; this is for a caller that means to hold the transpose.
     *
     * On the seam rather than beside the other whole-matrix operations because a library has its own routine
     * for it, `omatcopy` in the BLAS-like extensions, where the standard has none.
     */
    public fun transpose(a: DenseMatrix): DenseMatrix

    /** `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *  `beta == 0.0` overwrites [c] without reading it. [workspace] reuses any needed transpose packing. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    )

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly
     * exposed as `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is
     * neither read nor written. `alpha == 0.0` does not read either input and `beta == 0.0` does not read
     * selected destination entries. If [c] shares either input buffer, [workspace] supplies reusable staging.
     */
    @Suppress("LongParameterList") // the BLAS gemmt signature plus optional scratch
    public fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * [gemm] over borrowed column-major panels. [c] must not overlap either input; disjoint panels may share
     * a backing buffer. Implementations must preserve each physical leading dimension without copying.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: StridedMatrixView,
        transposeA: Boolean,
        b: StridedMatrixView,
        transposeB: Boolean,
        beta: Double,
        c: StridedMatrixView,
    ) {
        val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
        require(!c.overlaps(a) && !c.overlaps(b)) { "gemm: destination overlaps an input view" }
        if (alpha == 0.0 || k == 0) {
            // Scale and stop, as the dense reference and the host adapter do. Running the sum instead would
            // let an infinite operand reach `alpha * sum` and write NaN where the answer is beta times C.
            for (j in 0 until n) {
                for (i in 0 until m) {
                    c[i, j] = when (beta) {
                        0.0 -> 0.0
                        1.0 -> c[i, j]
                        else -> beta * c[i, j]
                    }
                }
            }
            return
        }
        for (j in 0 until n) {
            for (i in 0 until m) {
                var sum = 0.0
                for (p in 0 until k) {
                    val av = if (transposeA) a[p, i] else a[i, p]
                    val bv = if (transposeB) b[j, p] else b[p, j]
                    sum += av * bv
                }
                c[i, j] = alpha * sum + when (beta) {
                    0.0 -> 0.0
                    1.0 -> c[i, j]
                    else -> beta * c[i, j]
                }
            }
        }
    }

    /** [gemm] over borrowed panels into a fresh owned matrix. */
    public fun gemm(a: StridedMatrixView, b: StridedMatrixView): DenseMatrix {
        val result = DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, result.asView())
        return result
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     * Only the [lower] or upper triangle is written; `beta == 0.0` overwrites it without reading.
     *
     * In the non-transposed form, a zero entry used as the rank-one multiplier is skipped before it can
     * multiply an infinity, following Netlib `dsyrk`; the transposed dot-product form evaluates that product.
     * Pass a [workspace] to reuse the packed panels and diagonal tile.
     */
    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included; `beta == 0.0` overwrites [y] without reading it. */
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
     *  [lower] triangle of [a] is read; `beta == 0.0` overwrites [c] without reading it. [workspace] reuses
     *  the packed operand panels and edge tile. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: Workspace? = null,
    )

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`), the dense form a backend can dispatch. The free `ger` accepts
     *  [VectorStorage] operands and takes a sparse fast path. */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix)

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. */
    public fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, lower: Boolean = true)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. */
    public fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, lower: Boolean = true)

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Writes only the [lower] or upper triangle; `beta == 0.0` overwrites it without reading.
     *
     * In the non-transposed form, a rank step is skipped only when both output-column coefficients are zero,
     * following Netlib `dsyr2k`; the transposed form evaluates both products. Pass a [workspace] to reuse the
     * packed panels and diagonal tile.
     */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature plus optional scratch
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     * `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out.
     *
     * The diagonal is divided by, not tested: `dtrsv` carries no `info` and reports nothing, so a singular
     * triangle yields infinities or NaNs and the caller who needs the distinction tests the diagonal first.
     * The sparse [com.eignex.koblas.sparse.SparseBlas.trsv] follows the same rule.
     */
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags
     *  follow [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right.
     *  [workspace] reuses portable staging. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
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
     *  [trsm]. [workspace] reuses portable staging. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    )
}
