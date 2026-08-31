@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The portable dense matrix routines, the semantic reference a native [F64Blas] is validated against.
 *
 * @param configured the kernels the inner loops use, or null to follow the [F64Context] default.
 */
internal class F64ReferenceBlas(private val configured: F64Kernels? = null) : F64Blas {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val kernels: F64Kernels get() = configured ?: koblas.kernels

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
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
        applyBeta(kernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        val ad = a.data
        val rows = a.rows
        if (!transpose) {
            for (j in 0 until a.cols) {
                axpyArithmetic(kernels, y, 0, alpha * x[j], ad, j * rows, rows)
            }
        } else {
            val quads = DoubleArray(4)
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

    override fun transpose(a: F64DenseMatrix): F64DenseMatrix {
        val t = F64DenseMatrix(a.cols, a.rows)
        transposeBlocked(a.data, a.rows, a.cols, t.data)
        return t
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod")
    override fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        applyBeta(kernels, cd, 0, cd.size, beta)
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        val kernels = kernels
        when {
            !transposeA -> blockedGemmUpdate(
                kernels, alpha, a.data, b.data, b.rows, transposeB, cd, m, n, k,
                skipZeroCoefficient = false,
            )

            !transposeB -> blockedTransposedLeftUpdate(
                kernels, alpha, a.data, 0, a.rows, b.data, 0, b.rows, cd, 0, m, m, n, k,
            )

            b.data.size <= a.data.size -> {
                val packedB = DoubleArray(k * n)
                transposeBlocked(b.data, b.rows, b.cols, packedB)
                blockedTransposedLeftUpdate(
                    kernels, alpha, a.data, 0, a.rows, packedB, 0, k, cd, 0, m, m, n, k,
                )
            }

            else -> {
                val packedA = DoubleArray(m * k)
                transposeBlocked(a.data, a.rows, a.cols, packedA)
                blockedGemmUpdate(
                    kernels, alpha, packedA, b.data, b.rows, true, cd, m, n, k,
                    skipZeroCoefficient = false,
                )
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    override fun syrk(
        alpha: Double,
        a: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        scaleTriangle(kernels, cd, n, beta, lower)
        if (alpha == 0.0 || n == 0 || k == 0) return
        if (!transpose) {
            blockedSyrkUpdate(kernels, alpha, a.data, cd, n, k, lower)
        } else {
            workspace.borrow(n * k) { packed ->
                transposeBlocked(a.data, a.rows, a.cols, packed)
                blockedSyrkUpdate(kernels, alpha, packed, cd, n, k, lower, guardZeroColumns = false)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
        applyBeta(kernels, y, 0, n, beta)
        if (alpha == 0.0) return
        symvAccumulate(alpha, a.data, n, x, 0, y, 0, lower)
    }

    /** Accumulates alpha times A times x into y for the symmetric `n×n` [ad], reading only the [lower] or
     *  upper triangle. */
    @Suppress("LongParameterList") // two buffers with offsets plus the triangle flag
    private fun symvAccumulate(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        lower: Boolean,
    ) {
        val kernels = kernels
        for (j in 0 until n) {
            val base = j + j * n
            val xj = alpha * x[xOff + j]
            val runOff = if (lower) j + 1 else 0
            val len = if (lower) n - j - 1 else j
            y[yOff + j] += xj * ad[base]
            axpyArithmetic(kernels, y, yOff + runOff, xj, ad, runOff + j * n, len)
            y[yOff + j] += alpha * kernels.dot(ad, runOff + j * n, x, xOff + runOff, len)
        }
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
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
        if (right) {
            blockedSymmRightUpdate(kernels, alpha, a.data, b.data, cd, b.rows, m, lower)
        } else {
            blockedSymmLeftUpdate(kernels, alpha, a.data, b.data, cd, m, b.cols, lower)
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix) {
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
    override fun syr(alpha: Double, x: F64VectorLike, a: F64DenseMatrix, lower: Boolean) {
        requireShape(a.rows == a.cols) { "syr: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "syr: x length ${x.size} != ${a.rows}" }
        if (alpha == 0.0) return
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
    override fun syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, a: F64DenseMatrix, lower: Boolean) {
        requireShape(a.rows == a.cols) { "syr2: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows && y.size == a.rows) {
            "syr2: operand lengths ${x.size} and ${y.size} must both be ${a.rows}"
        }
        if (alpha == 0.0) return
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
    private fun rankUpdateData(x: F64VectorLike): DoubleArray = when (x) {
        is F64DenseVector -> x.data
        else -> DoubleArray(x.size).also { copy(x, F64DenseVector.wrap(it)) }
    }

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     *  [transpose]. Writes only the [lower] or upper triangle. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature plus optional scratch
    override fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(b.rows == a.rows && b.cols == a.cols) {
            "syr2k: B is ${b.rows}x${b.cols}, expected ${a.rows}x${a.cols} to match A"
        }
        requireShape(c.rows == n && c.cols == n) { "syr2k: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        scaleTriangle(kernels, c.data, n, beta, lower)
        if (alpha == 0.0 || n == 0 || k == 0) return
        if (!transpose) {
            blockedSyr2kUpdate(kernels, alpha, a.data, b.data, c.data, n, k, lower)
        } else {
            workspace.borrow(n * k) { packedA ->
                workspace.borrow(n * k) { packedB ->
                    transposeBlocked(a.data, a.rows, a.cols, packedA)
                    transposeBlocked(b.data, b.rows, b.cols, packedB)
                    blockedSyr2kUpdate(
                        kernels, alpha, packedA, packedB, c.data, n, k, lower,
                        guardZeroColumns = false,
                    )
                }
            }
        }
    }

    /** Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     *  `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out. */
    override fun trsv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = true)

    /** Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`). Flags follow [trsv];
     *  the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = true)

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    override fun trmv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = false)

    /** `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = false)
}

/** Square transpose tile, small enough to keep source and destination working sets in L1 cache. */
private const val TRANSPOSE_TILE = 32
