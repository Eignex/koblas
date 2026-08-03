package com.eignex.koblas.hostblas

import com.eignex.koblas.Blas
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.Uplo
import com.eignex.koblas.Workspace
import com.eignex.koblas.dispatchThresholds
import com.eignex.koblas.hostblas.HostBlasCalls.LEFT
import com.eignex.koblas.hostblas.HostBlasCalls.LOWER
import com.eignex.koblas.hostblas.HostBlasCalls.NO_TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.RIGHT
import com.eignex.koblas.hostblas.HostBlasCalls.ROW_MAJOR
import com.eignex.koblas.hostblas.HostBlasCalls.TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.UPPER

/**
 * The host OpenBLAS as the JVM's [Blas] half, bound with `java.lang.foreign`.
 *
 * Only the level-3 routines are native. `gemv` and `symv` delegate to the portable kernels, which are
 * Vector API SIMD on this platform and beat a foreign call outright: measured, `cblas_dgemv` lost to them
 * by 3x to 15x, because `O(n^2)` work over `O(n^2)` data has nothing to amortize a call against. The
 * triangular routines keep their portable defaults for the same reason — `cblas_dtrsv` at n=256 took
 * 65-136us against 25us portable.
 */
class HostBlas internal constructor() : Blas {
    override val name: String get() = "openblas"

    /** Above the reference (0) and the native dlopen backend (90), being the strongest JVM option. */
    override val priority: Int get() = 100

    /**
     * Delegated to [ReferenceLinearAlgebra]'s SIMD kernels rather than `cblas_dgemv`: level 2 does
     * `O(n²)` work over `O(n²)` data, so the per-call cost of handing a `double[]` to the native side
     * dominates and never amortizes. Measured across `n` 16..2048 the portable path wins everywhere,
     * by 3-4x at `n` 256 and an order of magnitude at 2048, where the native call manages about
     * 2 GB/s against the SIMD kernel's 20+. Level 3 and the factorizations amortize the same cost over
     * `O(n³)` work and stay native.
     */
    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) = ReferenceLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)

    @Suppress("LongParameterList") // the BLAS dgemm signature
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
        require(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        require(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
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
            ROW_MAJOR, transA, transB, m, n, k, alpha,
            HostBlasCalls.seg(a.data), a.cols, HostBlasCalls.seg(b.data), b.cols,
            beta, HostBlasCalls.seg(c.data), c.cols,
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
        require(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (minOf(n, k) < dispatchThresholds.level3) {
            ReferenceLinearAlgebra.syrk(alpha, a, transpose, beta, c, uplo, workspace)
            return
        }
        if (alpha == 0.0 || k == 0) {
            scaleUplo(c.data, n, beta, uplo)
            return
        }
        if (n == 0) return
        val trans = if (transpose) TRANS else NO_TRANS
        if (uplo != Uplo.FULL) {
            // Strict dsyrk semantics: one triangle written and beta-scaled, the other untouched.
            val u = if (uplo == Uplo.LOWER) LOWER else UPPER
            HostBlasCalls.dsyrk.invokeWithArguments(
                ROW_MAJOR, u, trans, n, k, alpha,
                HostBlasCalls.seg(a.data), a.cols, beta, HostBlasCalls.seg(c.data), n,
            )
            return
        }
        // dsyrk touches one triangle only, while the FULL contract promises the full, exactly
        // symmetric alpha term on top of a beta scale of all of C. Compute the alpha term into a
        // scratch lower triangle, mirror it, then combine.
        val w = workspace?.take(n * n) ?: DoubleArray(n * n)
        HostBlasCalls.dsyrk.invokeWithArguments(
            ROW_MAJOR, LOWER, trans, n, k, alpha,
            HostBlasCalls.seg(a.data), a.cols, 0.0, HostBlasCalls.seg(w), n,
        )
        for (i in 0 until n) {
            for (j in 0 until i) w[j * n + i] = w[i * n + j]
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

    /** Delegated to the SIMD kernels for the same reason as [gemv]; measured 1.5-2x faster up to
     *  `n` 1024 and 10x at 2048. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) =
        ReferenceLinearAlgebra.symv(alpha, a, x, beta, y, lower)

    @Suppress("LongParameterList") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        require(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
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
            ROW_MAJOR, side, uplo, c.rows, c.cols, alpha,
            HostBlasCalls.seg(a.data), m, HostBlasCalls.seg(b.data), c.cols,
            beta, HostBlasCalls.seg(c.data), c.cols,
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

    /** `beta` scale of the region [uplo] selects, honoring the `beta == 0` overwrite convention. */
    private fun scaleUplo(v: DoubleArray, n: Int, beta: Double, uplo: Uplo) {
        if (uplo == Uplo.FULL) {
            scaleInPlace(v, beta)
            return
        }
        if (beta == 1.0) return
        for (i in 0 until n) {
            val from = if (uplo == Uplo.LOWER) i * n else i * n + i
            val until = if (uplo == Uplo.LOWER) i * n + i + 1 else (i + 1) * n
            if (beta == 0.0) {
                v.fill(0.0, from, until)
            } else {
                for (idx in from until until) v[idx] *= beta
            }
        }
    }
}
