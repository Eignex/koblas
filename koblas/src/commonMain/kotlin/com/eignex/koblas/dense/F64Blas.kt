@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.*

/** Dense matrix routines as a backend half. */
public interface F64Blas : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val kernels: F64Kernels get() = koblas.kernels

    /** `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
     *  `beta == 0.0` overwrites [y] without reading it. */
    public fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    )

    /** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
    public fun gemv(a: F64DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /**
     * [gemv] over borrowed strided storage. The destination must not overlap [a] or [x]; disjoint views may
     * share one backing buffer. Implementations may pass offsets, strides, and leading dimensions directly
     * to BLAS and must not materialize a contiguous copy.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    public fun gemv(
        alpha: Double,
        a: F64StridedMatrixView,
        x: F64StridedVectorView,
        beta: Double,
        y: F64StridedVectorView,
        transpose: Boolean = false,
    ) {
        val xLength = if (transpose) a.rows else a.cols
        val yLength = if (transpose) a.cols else a.rows
        requireShape(x.size == xLength) { "gemv: x length ${x.size} != $xLength" }
        requireShape(y.size == yLength) { "gemv: y length ${y.size} != $yLength" }
        require(!y.overlaps(x) && !a.overlaps(y)) { "gemv: destination overlaps an input view" }
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
                val multiplier = alpha * x[j]
                if (multiplier != 0.0) for (i in 0 until a.rows) y[i] += multiplier * a[i, j]
            }
        }
    }

    /** [gemv] over borrowed storage into a fresh owned array. */
    public fun gemv(a: F64StridedMatrixView, x: F64StridedVectorView, transpose: Boolean = false): DoubleArray {
        val result = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, F64StridedVectorView(result, 0, result.size), transpose)
        return result
    }

    /**
     * Fresh transposed [a]. For a product prefer the transpose flags on [gemv] and [gemm], which read the
     * original storage without copying; this is for a caller that means to hold the transpose.
     *
     * On the seam rather than beside the other whole-matrix operations because a library has its own routine
     * for it, `omatcopy` in the BLAS-like extensions, where the standard has none.
     */
    public fun transpose(a: F64DenseMatrix): F64DenseMatrix

    /** `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *  `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: F64DenseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix(a.rows, b.cols)
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
        a: F64StridedMatrixView,
        transposeA: Boolean,
        b: F64StridedMatrixView,
        transposeB: Boolean,
        beta: Double,
        c: F64StridedMatrixView,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val otherK = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == otherK) { "gemm: op(A) is ${m}x$k but op(B) is ${otherK}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        require(!c.overlaps(a) && !c.overlaps(b)) { "gemm: destination overlaps an input view" }
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
    public fun gemm(a: F64StridedMatrixView, b: F64StridedMatrixView): F64DenseMatrix {
        val result = F64DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, result.asView())
        return result
    }

    /** `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     *  Only the [lower] or upper triangle is written; `beta == 0.0` overwrites it without reading. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    public fun syrk(
        alpha: Double,
        a: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included; `beta == 0.0` overwrites [y] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    public fun symv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
    )

    /** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     *  [lower] triangle of [a] is read; `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    )

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`), the dense form a backend can dispatch. The free `ger` accepts
     *  [F64VectorStorage] operands and takes a sparse fast path. */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix)

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. */
    public fun syr(alpha: Double, x: F64VectorLike, a: F64DenseMatrix, lower: Boolean = true)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. */
    public fun syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, a: F64DenseMatrix, lower: Boolean = true)

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Writes only the [lower] or upper triangle.
     *
     * A non-transposed pair is transposed into scratch first, so pass a [workspace] to keep a loop over this
     * routine from allocating `2·n·k` doubles per call. [syrk] borrows the same way for its one operand.
     */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature plus optional scratch
    public fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     * `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out.
     *
     * The diagonal is divided by, not tested: `dtrsv` carries no `info` and reports nothing, so a singular
     * triangle yields infinities or NaNs and the caller who needs the distinction tests the diagonal first.
     * That is the convention rather than a cost, and [trtri] shows it: having an `info` to return, it throws
     * on a zero diagonal. The sparse [com.eignex.koblas.sparse.F64SparseBlas.trsv] throws too, having no
     * BLAS routine whose silence it has to match.
     */
    public fun trsv(
        a: F64DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags
     *  follow [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    )

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    public fun trmv(
        a: F64DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of
     *  [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    )
}
