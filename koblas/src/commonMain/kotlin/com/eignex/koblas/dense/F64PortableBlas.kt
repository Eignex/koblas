@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.VectorLike
import com.eignex.koblas.internal.backend.BackendNames
import kotlin.math.abs

/**
 * The portable dense matrix routines, the semantic reference a native [Blas] is validated against.
 *
 * @param configured the kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
internal class F64PortableBlas(private val configured: Kernels? = null) : Blas {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val kernels: Kernels get() = configured ?: koblas.kernels

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        if (a.rows == 0 || a.cols == 0) return
        applyBeta(kernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        val kernels = kernels
        val ad = a.data
        val rows = a.rows
        if (!transpose) {
            var j = 0
            val bound = a.cols - 3
            while (j < bound) {
                kernels.axpy4(
                    y, 0, ad, j * rows, rows,
                    alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3], rows,
                )
                j += 4
            }
            while (j < a.cols) {
                axpyArithmetic(kernels, y, 0, alpha * x[j], ad, j * rows, rows)
                j++
            }
        } else {
            workspace.borrow(4) { quads ->
                var j = 0
                val bound = a.cols - 3
                while (j < bound) {
                    kernels.dot4(ad, j * rows, rows, x, 0, rows, quads, 0)
                    y[j] += alpha * quads[0]
                    y[j + 1] += alpha * quads[1]
                    y[j + 2] += alpha * quads[2]
                    y[j + 3] += alpha * quads[3]
                    j += 4
                }
                while (j < a.cols) {
                    y[j] += alpha * kernels.dot(ad, j * rows, x, 0, rows)
                    j++
                }
            }
        }
    }

    override fun transpose(a: DenseMatrix): DenseMatrix {
        val t = DenseMatrix(a.cols, a.rows)
        transposeBlocked(a.data, a.rows, a.cols, t.data)
        return t
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
        val cd = c.data
        applyBeta(kernels, cd, 0, cd.size, beta)
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        // One product for every shape and every target. The four transpositions differ only in how the
        // operands are read while they are packed, which is inside the packing rather than a path of its
        // own, and the only thing that varies by target is the tile the kernels supply.
        packedGemm(
            kernels, alpha, a.data, a.rows, transposeA, b.data, b.rows, transposeB, cd, m, n, k, workspace,
        )
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val (n, k) = requireSyrkShape(a, transpose, c, "syrk")
        val cd = c.data
        if (alpha == 0.0 || n == 0 || k == 0) {
            scaleTriangle(kernels, cd, n, beta, lower)
            return
        }
        val ad = a.data
        if (ad === cd) {
            workspace.borrow(ad.size) { copy ->
                ad.copyInto(copy)
                syrkFrom(alpha, copy, a.rows, transpose, beta, cd, n, k, lower, workspace)
            }
            return
        }
        syrkFrom(alpha, ad, a.rows, transpose, beta, cd, n, k, lower, workspace)
    }

    /** Implements [syrk] after an aliased operand has been snapshotted, if necessary. */
    @Suppress("LongParameterList")
    private fun syrkFrom(
        alpha: Double,
        ad: DoubleArray,
        lda: Int,
        transpose: Boolean,
        beta: Double,
        cd: DoubleArray,
        n: Int,
        k: Int,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        scaleTriangle(kernels, cd, n, beta, lower)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val scalingOverflows = packingScaleOverflows(alpha, ad)
        // Netlib's non-transposed traversal skips a raw zero multiplier. This is observable when another
        // value in the rank-one column is already non-finite or becomes non-finite when alpha scales it.
        if (!transpose && (!alpha.isFinite() || scalingOverflows || ad.any { !it.isFinite() })) {
            blockedSyrkUpdate(kernels, alpha, ad, cd, n, k, lower)
            return
        }
        // The old transposed traversal did not skip zero multipliers, but it scaled the output column's
        // coefficient rather than the other factor. Retain that order when packing would overflow a finite
        // value; actual non-finite inputs continue through the packed path and keep zero-times-infinity NaNs.
        if (transpose && scalingOverflows) {
            workspace.borrowTransposed(ad, lda, n) { packed ->
                blockedSyrkUpdate(kernels, alpha, packed, cd, n, k, lower, guardZeroColumns = false)
            }
            return
        }
        packedTriangularGemm(
            kernels, alpha, ad, lda, transpose, ad, lda, !transpose, cd, n, k, lower, workspace,
        )
    }

    /** Whether packing would turn a finite entry non-finite by applying [alpha] before the tile product. */
    private fun packingScaleOverflows(alpha: Double, values: DoubleArray): Boolean {
        if (!alpha.isFinite() || alpha in -1.0..1.0) return false
        return values.any { it.isFinite() && !(alpha * it).isFinite() }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        val n = requireSymvShape(a, x.size, y.size)
        applyBeta(kernels, y, 0, n, beta)
        if (alpha == 0.0) return
        symvAccumulate(alpha, a.data, n, x, y, lower)
    }

    /** Accumulates alpha times A times x into y for the symmetric `n×n` [ad], reading only the [lower] or
     *  upper triangle. */
    private fun symvAccumulate(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        y: DoubleArray,
        lower: Boolean,
    ) {
        val kernels = kernels
        for (j in 0 until n) {
            val base = j + j * n
            val xj = alpha * x[j]
            val runOff = if (lower) j + 1 else 0
            val len = if (lower) n - j - 1 else j
            y[j] += xj * ad[base]
            y[j] += alpha * kernels.dotAxpy(y, runOff, xj, ad, runOff + j * n, x, runOff, len)
        }
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSquare(a, "symm")
        val m = a.rows
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        // Both sides check A against B before anything is written, since scaling C is part of the
        // operation and a call that cannot go through must not have performed half of it.
        if (right) {
            requireShape(b.cols == m) { "symm right: B has ${b.cols} cols, expected $m" }
        } else {
            requireShape(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        }
        val cd = c.data
        applyBeta(kernels, cd, 0, cd.size, beta)
        if (alpha == 0.0 || m == 0 || b.rows == 0 || b.cols == 0) return
        // The same packed product the general case uses. A symmetric operand differs only in which side of
        // the diagonal the packing reads each element from, so there is no separate implementation and no
        // separate kernel: C = alpha * A * B on the left, C = alpha * B * A on the right.
        if (right) {
            packedGemm(
                kernels, alpha, b.data, b.rows, false, a.data, m, false, cd,
                b.rows, m, m, workspace, symmetricB = lower,
            )
        } else {
            packedGemm(
                kernels, alpha, a.data, m, false, b.data, b.rows, false, cd,
                m, b.cols, m, workspace, symmetricA = lower,
            )
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0) return
        val kernels = kernels
        for (j in 0 until a.cols) {
            if (y[j] != 0.0) axpyArithmetic(kernels, a.data, a.colOffset(j), alpha * y[j], x, 0, a.rows)
        }
    }

    /**
     * `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle.
     *
     * Non-dense vectors are staged once so the rank update itself is a sequence of contiguous Level 1 calls.
     */
    override fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, lower: Boolean) {
        requireSyrShape(a, x.size, "syr")
        if (alpha == 0.0) return
        val kernels = kernels
        val n = a.rows
        val ad = a.data
        val xs = rankUpdateData(x)
        for (j in 0 until n) {
            if (xs[j] == 0.0) continue
            val xj = alpha * xs[j]
            val from = if (lower) j else 0
            val length = if (lower) n - j else j + 1
            axpyArithmetic(kernels, ad, from + j * n, xj, xs, from, length)
        }
    }

    /**
     * `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle.
     *
     * Non-dense operands are staged once so the rank update itself is a sequence of contiguous Level 1 calls.
     */
    override fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, lower: Boolean) {
        requireSyr2Shape(a, x.size, y.size, "syr2")
        if (alpha == 0.0) return
        val kernels = kernels
        val n = a.rows
        val ad = a.data
        val xs = rankUpdateData(x)
        val ys = rankUpdateData(y)
        for (j in 0 until n) {
            if (xs[j] == 0.0 && ys[j] == 0.0) continue
            val from = if (lower) j else 0
            val length = if (lower) n - j else j + 1
            val matrixOffset = from + j * n
            axpyArithmetic(kernels, ad, matrixOffset, alpha * ys[j], xs, from, length)
            axpyArithmetic(kernels, ad, matrixOffset, alpha * xs[j], ys, from, length)
        }
    }

    /** Returns contiguous rank-update operands; sparse copies use the registered sparse Level 1 scatter. */
    private fun rankUpdateData(x: VectorLike): DoubleArray = when (x) {
        is DenseVector -> x.data
        else -> DoubleArray(x.size).also { copy(x, DenseVector.wrap(it)) }
    }

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     *  [transpose]. Writes only the [lower] or upper triangle. */
    @Suppress("LongParameterList", "ReturnCount") // the BLAS dsyr2k signature plus scratch; alias guards
    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val (n, k) = requireSyr2kShape(a, b, transpose, c, "syr2k")
        val cd = c.data
        if (alpha == 0.0 || n == 0 || k == 0) {
            scaleTriangle(kernels, cd, n, beta, lower)
            return
        }
        val ad = a.data
        val bd = b.data
        if (ad === cd) {
            workspace.borrow(ad.size) { copyA ->
                ad.copyInto(copyA)
                if (bd === cd) {
                    syr2kFrom(alpha, copyA, a.rows, copyA, b.rows, transpose, beta, cd, n, k, lower, workspace)
                } else {
                    syr2kFrom(alpha, copyA, a.rows, bd, b.rows, transpose, beta, cd, n, k, lower, workspace)
                }
            }
            return
        }
        if (bd === cd) {
            workspace.borrow(bd.size) { copyB ->
                bd.copyInto(copyB)
                syr2kFrom(alpha, ad, a.rows, copyB, b.rows, transpose, beta, cd, n, k, lower, workspace)
            }
            return
        }
        syr2kFrom(alpha, ad, a.rows, bd, b.rows, transpose, beta, cd, n, k, lower, workspace)
    }

    /** Implements [syr2k] after aliased operands have been snapshotted, if necessary. */
    @Suppress("LongParameterList")
    private fun syr2kFrom(
        alpha: Double,
        ad: DoubleArray,
        lda: Int,
        bd: DoubleArray,
        ldb: Int,
        transpose: Boolean,
        beta: Double,
        cd: DoubleArray,
        n: Int,
        k: Int,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        scaleTriangle(kernels, cd, n, beta, lower)
        // The retained traversal scales the output-column coefficient of each cross-product and skips only
        // when both raw coefficients are zero. Moving alpha to the packed row factor changes exceptional
        // arithmetic, so preserve the old evaluation order whenever a value is non-finite or scaling a
        // finite value would overflow.
        val exceptional = packedSyr2kChangesExceptionalArithmetic(alpha, ad, bd, beta, cd, n, k, lower)
        if (exceptional) {
            if (!transpose) {
                blockedSyr2kUpdate(kernels, alpha, ad, bd, cd, n, k, lower)
            } else {
                workspace.borrowTransposed(ad, lda, n) { packedA ->
                    workspace.borrowTransposed(bd, ldb, n) { packedB ->
                        blockedSyr2kUpdate(
                            kernels, alpha, packedA, packedB, cd, n, k, lower,
                            guardZeroColumns = false,
                        )
                    }
                }
            }
            return
        }
        packedTriangularGemm(
            kernels, alpha, ad, lda, transpose, bd, ldb, !transpose, cd, n, k, lower, workspace,
        )
        packedTriangularGemm(
            kernels, alpha, bd, ldb, transpose, ad, lda, !transpose, cd, n, k, lower, workspace,
        )
    }

    /** Whether separating the two cross-products can differ from the retained rank-2k evaluation order. */
    @Suppress("LongParameterList")
    private fun packedSyr2kChangesExceptionalArithmetic(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        beta: Double,
        c: DoubleArray,
        n: Int,
        k: Int,
        lower: Boolean,
    ): Boolean {
        if (!alpha.isFinite()) return true
        val scalingCanOverflow = alpha !in -1.0..1.0
        var maxA = 0.0
        for (value in a) {
            if (!value.isFinite() || (scalingCanOverflow && !(alpha * value).isFinite())) return true
            maxA = maxOf(maxA, abs(value))
        }
        var maxB = 0.0
        for (value in b) {
            if (!value.isFinite() || (scalingCanOverflow && !(alpha * value).isFinite())) return true
            maxB = maxOf(maxB, abs(value))
        }
        if (maxA == 0.0 || maxB == 0.0) return false

        var maxC = 0.0
        if (beta != 0.0) {
            for (j in 0 until n) {
                val from = if (lower) j else 0
                val until = if (lower) n else j + 1
                for (i in from until until) {
                    val value = c[i + j * n]
                    if (!value.isFinite()) return true
                    maxC = maxOf(maxC, abs(value))
                }
            }
        }

        // Each packed call accumulates one cross-product before the other is added. An absolute bound on
        // both products plus the scaled destination keeps every packed intermediate finite; otherwise the
        // retained loop must interleave them rank by rank so opposite infinities do not manufacture NaN.
        val productLimit = (Double.MAX_VALUE - maxC) / (2.0 * k)
        return productExceeds(productLimit, abs(alpha), maxA, maxB)
    }

    /** Compares three positive finite factors with [limit] without overflowing the comparison itself. */
    private fun productExceeds(limit: Double, first: Double, second: Double, third: Double): Boolean {
        if (first == 0.0 || second == 0.0 || third == 0.0) return false
        var largest = first
        var middle = second
        var smallest = third
        if (largest < middle) {
            val swap = largest
            largest = middle
            middle = swap
        }
        if (middle < smallest) {
            val swap = middle
            middle = smallest
            smallest = swap
        }
        if (largest < middle) {
            val swap = largest
            largest = middle
            middle = swap
        }
        return limit / largest / middle / smallest < 1.0
    }

    /** Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     *  `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out. */
    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = true)

    /** Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`). Flags follow [trsv];
     *  the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = true, workspace = workspace)

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = false)

    /** `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = false, workspace = workspace)
}
