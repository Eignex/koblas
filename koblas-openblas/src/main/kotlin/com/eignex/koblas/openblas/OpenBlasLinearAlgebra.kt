package com.eignex.koblas.openblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LdlDecomposition
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.QrDecomposition
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.Uplo
import com.eignex.koblas.Workspace
import com.eignex.koblas.openblas.OpenBlasCalls.LEFT
import com.eignex.koblas.openblas.OpenBlasCalls.LOWER
import com.eignex.koblas.openblas.OpenBlasCalls.NON_UNIT
import com.eignex.koblas.openblas.OpenBlasCalls.NO_TRANS
import com.eignex.koblas.openblas.OpenBlasCalls.RIGHT
import com.eignex.koblas.openblas.OpenBlasCalls.ROW_MAJOR
import com.eignex.koblas.openblas.OpenBlasCalls.TRANS
import com.eignex.koblas.openblas.OpenBlasCalls.UNIT
import com.eignex.koblas.openblas.OpenBlasCalls.UPPER

/**
 * [LinearAlgebra] backed by OpenBLAS through the Bytedeco JavaCPP presets. The natives ship in the
 * org.bytedeco:openblas artifacts and load on first use; no system BLAS installation is required.
 *
 * Not every routine goes native. Handing a `double[]` to the native side costs per call, which `O(n³)`
 * work amortizes and `O(n²)` work does not, so the level-2 products ([gemv], [symv]) delegate to
 * [ReferenceLinearAlgebra]'s SIMD kernels while level 3 and the factorizations dispatch to OpenBLAS.
 * Each routine's KDoc says which it is, with the measurement behind it.
 *
 * Calls go through `java.lang.foreign` downcalls declared in [OpenBlasCalls], with heap access enabled,
 * so a `DoubleArray` is pinned for the call rather than copied. Koblas storage is row-major, which CBLAS
 * and LAPACKE support directly, so buffers cross the boundary without repacking or copying. Semantics match
 * [com.eignex.koblas.ReferenceLinearAlgebra] exactly as
 * specified by the [LinearAlgebra] contract: `beta == 0` overwrites without reading, `alpha == 0`
 * reduces to the `beta` scale, [syrk] produces the full, exactly symmetric result, and [factor]
 * packs `L`\`U` in the shared [LuDecomposition] format so factorizations interchange between backends.
 *
 * OpenBLAS runs single-threaded by default here: its threaded LAPACK path overflows default-sized JVM
 * thread stacks (SIGSEGV) and loses badly to the single-threaded path at moderate sizes anyway. To opt
 * into OpenBLAS threading set the `koblas.openblas.threads` system property (before first use) or the
 * `OPENBLAS_NUM_THREADS` environment variable, and give calling threads a large stack (`-Xss16m`).
 *
 * Registered as a `ServiceLoader` provider for [LinearAlgebra]: putting this artifact on the classpath
 * activates it via `platformLinearAlgebra()` with no code changes.
 */
class OpenBlasLinearAlgebra : LinearAlgebra {
    /** Applies the threading default once, when the first instance loads the class. */
    companion object {
        init {
            val requested = System.getProperty("koblas.openblas.threads")?.toIntOrNull()
            when {
                requested != null -> OpenBlasCalls.setThreads(requested)

                // An explicit environment override is already honored by OpenBLAS itself.
                System.getenv("OPENBLAS_NUM_THREADS") == null -> OpenBlasCalls.setThreads(1)
            }
        }
    }

    override val name: String get() = "openblas"

    /** The bundled-natives backend outranks every other automatic candidate. */
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
        val transA = if (transposeA) TRANS else NO_TRANS
        val transB = if (transposeB) TRANS else NO_TRANS
        OpenBlasCalls.dgemm.invokeWithArguments(
            ROW_MAJOR, transA, transB, m, n, k, alpha,
            OpenBlasCalls.seg(a.data), a.cols, OpenBlasCalls.seg(b.data), b.cols,
            beta, OpenBlasCalls.seg(c.data), c.cols,
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
        if (alpha == 0.0 || k == 0) {
            scaleUplo(c.data, n, beta, uplo)
            return
        }
        if (n == 0) return
        val trans = if (transpose) TRANS else NO_TRANS
        if (uplo != Uplo.FULL) {
            // Strict dsyrk semantics: one triangle written and beta-scaled, the other untouched.
            val u = if (uplo == Uplo.LOWER) LOWER else UPPER
            OpenBlasCalls.dsyrk.invokeWithArguments(
                ROW_MAJOR, u, trans, n, k, alpha,
                OpenBlasCalls.seg(a.data), a.cols, beta, OpenBlasCalls.seg(c.data), n,
            )
            return
        }
        // dsyrk touches one triangle only, while the FULL contract promises the full, exactly
        // symmetric alpha term on top of a beta scale of all of C. Compute the alpha term into a
        // scratch lower triangle, mirror it, then combine.
        val w = workspace?.take(n * n) ?: DoubleArray(n * n)
        OpenBlasCalls.dsyrk.invokeWithArguments(
            ROW_MAJOR, LOWER, trans, n, k, alpha,
            OpenBlasCalls.seg(a.data), a.cols, 0.0, OpenBlasCalls.seg(w), n,
        )
        for (i in 0 until n) {
            for (j in 0 until i) w[j * n + i] = w[i * n + j]
        }
        val cd = c.data
        when (beta) {
            0.0 -> w.copyInto(cd)

            else -> {
                if (beta != 1.0) OpenBlasCalls.dscal.invokeWithArguments(cd.size, beta, OpenBlasCalls.seg(cd), 1)
                OpenBlasCalls.daxpy.invokeWithArguments(
                    cd.size,
                    1.0,
                    OpenBlasCalls.seg(w),
                    1,
                    OpenBlasCalls.seg(cd),
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
        OpenBlasCalls.dsymm.invokeWithArguments(
            ROW_MAJOR, side, uplo, c.rows, c.cols, alpha,
            OpenBlasCalls.seg(a.data), m, OpenBlasCalls.seg(b.data), c.cols,
            beta, OpenBlasCalls.seg(c.data), c.cols,
        )
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv, singular = false)
        val ipiv = IntArray(n)
        val info = OpenBlasCalls.dgetrf.invokeWithArguments(
            ROW_MAJOR,
            n,
            n,
            OpenBlasCalls.seg(lu),
            n,
            OpenBlasCalls.seg(ipiv),
        ) as Int
        check(info >= 0) { "dgetrf: illegal argument ${-info}" }
        // dgetrf reports successive row swaps (1-based); replay them to get the permutation form
        // LuDecomposition uses (piv[k] = original row now at position k).
        for (k in 0 until n) {
            val p = ipiv[k] - 1
            if (p != k) {
                val t = piv[k]
                piv[k] = piv[p]
                piv[p] = t
            }
        }
        return LuDecomposition(n, lu, piv, singular = info > 0)
    }

    /**
     * Delegated to [ReferenceLinearAlgebra] for the same reason as [gemv]: two triangular solves over
     * the `n²` factor are `O(n²)` work on `O(n²)` data, so the per-call cost dominates. Measured at
     * `n` 256 the portable path takes 25 us against 65-136 us through `cblas_dtrsv`. The blocked
     * [solve] below keeps `dtrsm`, which amortizes the same cost across many right-hand sides.
     */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray = ReferenceLinearAlgebra.solveInto(lu, b, out, transpose, workspace)

    override fun solve(lu: LuDecomposition, b: DenseMatrix, transpose: Boolean): DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        if (n == 0 || nrhs == 0) return DenseMatrix.wrap(n, nrhs, DoubleArray(n * nrhs))
        // One or two columns do not cover the native call's cost: at n 256 dtrsm needs 100-140 us for a
        // single column against the delegated path's 25 us. From four columns on, dtrsm is ahead.
        if (nrhs < NATIVE_TRSM_MIN_RHS) {
            val out = DoubleArray(n * nrhs)
            val col = DoubleArray(n)
            for (c in 0 until nrhs) {
                for (i in 0 until n) col[i] = b.data[i * nrhs + c]
                val x = solve(lu, col, transpose)
                for (i in 0 until n) out[i * nrhs + c] = x[i]
            }
            return DenseMatrix.wrap(n, nrhs, out)
        }
        val f = lu.lu
        // Permute plus two block triangular solves; dtrsm is row-major native, so no LAPACKE
        // transposition tax scales with nrhs.
        return if (transpose) {
            val y = b.data.copyOf()
            OpenBlasCalls.dtrsm.invokeWithArguments(
                ROW_MAJOR, LEFT, UPPER, TRANS, NON_UNIT, n, nrhs, 1.0,
                OpenBlasCalls.seg(f), n, OpenBlasCalls.seg(y), nrhs,
            )
            OpenBlasCalls.dtrsm.invokeWithArguments(
                ROW_MAJOR, LEFT, LOWER, TRANS, UNIT, n, nrhs, 1.0,
                OpenBlasCalls.seg(f), n, OpenBlasCalls.seg(y), nrhs,
            )
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) y.copyInto(x, lu.piv[i] * nrhs, i * nrhs, (i + 1) * nrhs)
            DenseMatrix.wrap(n, nrhs, x)
        } else {
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) b.data.copyInto(x, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
            OpenBlasCalls.dtrsm.invokeWithArguments(
                ROW_MAJOR, LEFT, LOWER, NO_TRANS, UNIT, n, nrhs, 1.0,
                OpenBlasCalls.seg(f), n, OpenBlasCalls.seg(x), nrhs,
            )
            OpenBlasCalls.dtrsm.invokeWithArguments(
                ROW_MAJOR, LEFT, UPPER, NO_TRANS, NON_UNIT, n, nrhs, 1.0,
                OpenBlasCalls.seg(f), n, OpenBlasCalls.seg(x), nrhs,
            )
            DenseMatrix.wrap(n, nrhs, x)
        }
    }

    override fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        val x = b.data.copyOf()
        if (n == 0 || nrhs == 0) return DenseMatrix.wrap(n, nrhs, x)
        val info = OpenBlasCalls.dsytrs.invokeWithArguments(
            ROW_MAJOR, 'L'.code.toByte(), n, nrhs,
            OpenBlasCalls.seg(ldl.ldl), n, OpenBlasCalls.seg(ldl.ipiv), OpenBlasCalls.seg(x), nrhs,
        ) as Int
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return DenseMatrix.wrap(n, nrhs, x)
    }

    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv, singular = false)
        val info = OpenBlasCalls.dsytrf.invokeWithArguments(
            ROW_MAJOR,
            'L'.code.toByte(),
            n,
            OpenBlasCalls.seg(buf),
            n,
            OpenBlasCalls.seg(ipiv),
        ) as Int
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, singular = info > 0)
    }

    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
        if (n == 0) return out
        val x = out
        if (out !== b) b.copyInto(out)
        val info = OpenBlasCalls.dsytrs.invokeWithArguments(
            ROW_MAJOR, 'L'.code.toByte(), n, 1,
            OpenBlasCalls.seg(ldl.ldl), n, OpenBlasCalls.seg(ldl.ipiv), OpenBlasCalls.seg(x), 1,
        ) as Int
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return x
    }

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = OpenBlasCalls.dgeqrf.invokeWithArguments(
                ROW_MAJOR,
                m,
                n,
                OpenBlasCalls.seg(buf),
                n,
                OpenBlasCalls.seg(tau),
            ) as Int
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    override fun applyQ(qr: QrDecomposition, y: DoubleArray, transpose: Boolean): DoubleArray {
        require(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        val c = y.copyOf()
        if (qr.tau.isEmpty()) return c
        val side = 'L'.code.toByte()
        val trans = (if (transpose) 'T' else 'N').code.toByte()
        val info = OpenBlasCalls.dormqr.invokeWithArguments(
            ROW_MAJOR, side, trans, qr.m, 1, qr.tau.size,
            OpenBlasCalls.seg(qr.qr), qr.n, OpenBlasCalls.seg(qr.tau), OpenBlasCalls.seg(c), 1,
        ) as Int
        check(info == 0) { "dormqr: illegal argument ${-info}" }
        return c
    }

    override fun rcond(lu: LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val out = DoubleArray(1)
        val info = OpenBlasCalls.dgecon.invokeWithArguments(
            ROW_MAJOR,
            '1'.code.toByte(),
            n,
            OpenBlasCalls.seg(lu.lu),
            n,
            anorm,
            OpenBlasCalls.seg(out),
        ) as Int
        check(info == 0) { "dgecon: illegal argument ${-info}" }
        return out[0]
    }

    /** `v = beta * v` honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)

            beta != 1.0 && v.isNotEmpty() -> OpenBlasCalls.dscal.invokeWithArguments(
                v.size,
                beta,
                OpenBlasCalls.seg(v),
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

/** Right-hand-side count from which `dtrsm` beats the delegated per-column solves. */
private const val NATIVE_TRSM_MIN_RHS = 4
