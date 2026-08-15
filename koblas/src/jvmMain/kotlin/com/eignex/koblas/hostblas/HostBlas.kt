package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DispatchThresholds
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.scaleUplo
import com.eignex.koblas.dense.trmm
import com.eignex.koblas.dense.trmv
import com.eignex.koblas.dense.trsm
import com.eignex.koblas.dense.trsv
import com.eignex.koblas.dispatchThresholds
import com.eignex.koblas.hostblas.HostBlasCalls.COL_MAJOR
import com.eignex.koblas.hostblas.HostBlasCalls.LEFT
import com.eignex.koblas.hostblas.HostBlasCalls.LOWER
import com.eignex.koblas.hostblas.HostBlasCalls.NO_TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.RIGHT
import com.eignex.koblas.hostblas.HostBlasCalls.TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.UPPER
import com.eignex.koblas.hostblas.HostBlasCalls.seg
import com.eignex.koblas.requireShape

/** Dense matrix routines backed by a host OpenBLAS. */
public class HostBlas internal constructor() : Blas {
    override val name: String get() = "openblas"

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** Portable below the level-2 gate, cblas_dger above it. */
    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        if (minOf(a.rows, a.cols) < dispatchThresholds.level2) return super.ger(alpha, x, y, a)
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        HostBlasCalls.dger.invokeWithArguments(
            COL_MAJOR, a.rows, a.cols, alpha, seg(x), 1, seg(y), 1, seg(a.data), a.rows,
        )
    }

    /** Portable below the level-2 gate, cblas_dtrsv above it. */
    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows < dispatchThresholds.level2) return super.trsv(a, x, lower, transpose, unitDiag)
        requireShape(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        if (a.rows == 0) return
        HostBlasCalls.dtrsv.invokeWithArguments(
            COL_MAJOR, uploOf(lower), transOf(transpose), diagOf(unitDiag), a.rows,
            seg(a.data), a.rows, seg(x), 1,
        )
    }

    /** Portable below the level-2 gate, cblas_dtrmv above it. */
    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows < dispatchThresholds.level2) return super.trmv(a, x, lower, transpose, unitDiag)
        requireShape(a.rows == a.cols) { "trmv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        if (a.rows == 0) return
        HostBlasCalls.dtrmv.invokeWithArguments(
            COL_MAJOR, uploOf(lower), transOf(transpose), diagOf(unitDiag), a.rows,
            seg(a.data), a.rows, seg(x), 1,
        )
    }

    /** Portable below the level-3 gate, cblas_dtrsm above it. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
    ) {
        if (minOf(a.rows, b.rows, b.cols) < dispatchThresholds.level3) {
            return super.trsm(a, b, lower, transpose, unitDiag, right)
        }
        triangularSolveOrMultiply(a, b, lower, transpose, unitDiag, right, solve = true)
    }

    /** Portable below the level-3 gate, cblas_dtrmm above it. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
    ) {
        if (minOf(a.rows, b.rows, b.cols) < dispatchThresholds.level3) {
            return super.trmm(a, b, lower, transpose, unitDiag, right)
        }
        triangularSolveOrMultiply(a, b, lower, transpose, unitDiag, right, solve = false)
    }

    /** dtrsm and dtrmm take the same arguments and differ only in the entry point. koblas fixes alpha at 1. */
    @Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
    private fun triangularSolveOrMultiply(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        solve: Boolean,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireShape(a.rows == a.cols) { "$what requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            requireShape(b.cols == a.rows) { "$what right: B has ${b.cols} cols, expected ${a.rows}" }
        } else {
            requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
        }
        if (a.rows == 0 || b.rows == 0 || b.cols == 0) return
        val handle = if (solve) HostBlasCalls.dtrsm else HostBlasCalls.dtrmm
        handle.invokeWithArguments(
            COL_MAJOR, if (right) RIGHT else LEFT, uploOf(lower), transOf(transpose), diagOf(unitDiag),
            b.rows, b.cols, 1.0, seg(a.data), a.rows, seg(b.data), b.rows,
        )
    }

    private fun uploOf(lower: Boolean) = if (lower) LOWER else UPPER
    private fun transOf(transpose: Boolean) = if (transpose) TRANS else NO_TRANS
    private fun diagOf(unitDiag: Boolean) = if (unitDiag) HostBlasCalls.UNIT else HostBlasCalls.NON_UNIT

    /** Portable below [DispatchThresholds.level2], native above it. */
    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        if (minOf(a.rows, a.cols) < dispatchThresholds.level2) {
            ReferenceLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)
            return
        }
        val trans = if (transpose) TRANS else NO_TRANS
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        if (alpha == 0.0 || xLen == 0) {
            scaleInPlace(y, beta)
            return
        }
        if (yLen == 0) return
        HostBlasCalls.dgemv.invokeWithArguments(
            COL_MAJOR, trans, a.rows, a.cols, alpha,
            seg(a.data), a.rows, seg(x), 1, beta, seg(y), 1,
        )
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        if (alpha == 0.0 || k == 0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (m == 0 || n == 0) return
        if (minOf(m, n, k) < dispatchThresholds.level3) {
            ReferenceLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c)
            return
        }
        val transA = if (transposeA) TRANS else NO_TRANS
        val transB = if (transposeB) TRANS else NO_TRANS
        HostBlasCalls.dgemm.invokeWithArguments(
            COL_MAJOR, transA, transB, m, n, k, alpha,
            HostBlasCalls.seg(a.data), a.rows, HostBlasCalls.seg(b.data), b.rows,
            beta, HostBlasCalls.seg(c.data), c.rows,
        )
    }

    @Suppress("LongParameterList", "ReturnCount") // dsyrk's arguments plus scratch; guard-clause style
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (minOf(n, k) < dispatchThresholds.level3) {
            ReferenceLinearAlgebra.syrk(alpha, a, transpose, beta, c, uplo, workspace)
            return
        }
        if (alpha == 0.0 || k == 0) {
            scaleUplo(vectorKernels, c.data, n, beta, uplo)
            return
        }
        if (n == 0) return
        val trans = if (transpose) TRANS else NO_TRANS
        if (uplo != Uplo.FULL) {
            // Strict dsyrk semantics: one triangle written and beta-scaled, the other untouched.
            val u = if (uplo == Uplo.LOWER) LOWER else UPPER
            HostBlasCalls.dsyrk.invokeWithArguments(
                COL_MAJOR, u, trans, n, k, alpha,
                HostBlasCalls.seg(a.data), a.rows, beta, HostBlasCalls.seg(c.data), n,
            )
            return
        }
        // dsyrk touches one triangle only, so the FULL contract needs the alpha term computed into a
        // scratch lower triangle, mirrored, then combined with a beta scale of all of C.
        val w = workspace?.take(n * n) ?: DoubleArray(n * n)
        HostBlasCalls.dsyrk.invokeWithArguments(
            COL_MAJOR, LOWER, trans, n, k, alpha,
            HostBlasCalls.seg(a.data), a.rows, 0.0, HostBlasCalls.seg(w), n,
        )
        for (j in 0 until n) {
            for (i in j + 1 until n) w[j + i * n] = w[i + j * n]
        }
        val cd = c.data
        when (beta) {
            0.0 -> w.copyInto(cd)

            else -> {
                if (beta != 1.0) HostBlasCalls.dscal.invokeWithArguments(cd.size, beta, HostBlasCalls.seg(cd), 1)
                HostBlasCalls.daxpy.invokeWithArguments(
                    cd.size,
                    1.0,
                    HostBlasCalls.seg(w),
                    1,
                    HostBlasCalls.seg(cd),
                    1,
                )
            }
        }
    }

    /**
     * Portable below [DispatchThresholds.level2], native above it, as for [gemv]. The shape is checked ahead
     * of the gate: `dsymv` takes one dimension and a leading dimension, so a non-square matrix would have it
     * read `n²` entries from a shorter array, past the end of the pinned buffer.
     */
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        if (a.rows < dispatchThresholds.level2) {
            ReferenceLinearAlgebra.symv(alpha, a, x, beta, y, lower)
            return
        }
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
        if (alpha == 0.0 || n == 0) {
            scaleInPlace(y, beta)
            return
        }
        HostBlasCalls.dsymv.invokeWithArguments(
            COL_MAJOR, if (lower) LOWER else UPPER, n, alpha,
            seg(a.data), n, seg(x), 1, beta, seg(y), 1,
        )
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        if (minOf(a.rows, b.rows, b.cols) < dispatchThresholds.level3) {
            ReferenceLinearAlgebra.symm(alpha, a, b, beta, c, lower, right)
            return
        }
        val m = a.rows
        require(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        require((if (right) b.cols else b.rows) == m) {
            "symm: B is ${b.rows}x${b.cols}, expected dimension $m on the ${if (right) "cols" else "rows"} side"
        }
        if (alpha == 0.0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (c.rows == 0 || c.cols == 0) return
        val uplo = if (lower) LOWER else UPPER
        val side = if (right) RIGHT else LEFT
        HostBlasCalls.dsymm.invokeWithArguments(
            COL_MAJOR, side, uplo, c.rows, c.cols, alpha,
            HostBlasCalls.seg(a.data), m, HostBlasCalls.seg(b.data), b.rows,
            beta, HostBlasCalls.seg(c.data), c.rows,
        )
    }

    /** `v = beta * v` honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)

            beta != 1.0 && v.isNotEmpty() -> HostBlasCalls.dscal.invokeWithArguments(
                v.size,
                beta,
                HostBlasCalls.seg(v),
                1,
            )
        }
    }
}
