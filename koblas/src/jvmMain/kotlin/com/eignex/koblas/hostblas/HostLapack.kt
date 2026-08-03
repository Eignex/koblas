package com.eignex.koblas.hostblas

import com.eignex.koblas.Blas
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Lapack
import com.eignex.koblas.LdlDecomposition
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.QrDecomposition
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.Workspace
import com.eignex.koblas.dispatchThresholds
import com.eignex.koblas.hostblas.HostBlasCalls.LEFT
import com.eignex.koblas.hostblas.HostBlasCalls.LOWER
import com.eignex.koblas.hostblas.HostBlasCalls.NON_UNIT
import com.eignex.koblas.hostblas.HostBlasCalls.NO_TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.ROW_MAJOR
import com.eignex.koblas.hostblas.HostBlasCalls.TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.UNIT
import com.eignex.koblas.hostblas.HostBlasCalls.UPPER

/**
 * The host LAPACKE as the JVM's [Lapack] half.
 *
 * Native where the work is `O(n^3)`: the LU, LDL and QR factorizations, the blocked multi-RHS solve, and
 * the condition estimate. The single-vector solves delegate to the portable kernels, as on every other
 * platform.
 *
 * The Cholesky family keeps its portable defaults, which is a measurement rather than an omission: the
 * SIMD factorization beat single-threaded OpenBLAS at n=256 (696us against 1524us), matched it at 1024
 * (49.2ms against 51.4ms), and trailed only at 2048 (635ms against 326ms). Overriding it would make the
 * common sizes slower.
 */
class HostLapack internal constructor() : Lapack {
    override val name: String get() = "openblas"

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = 100

    override fun factor(a: DenseMatrix): LuDecomposition {
        if (a.rows < dispatchThresholds.lapack) return ReferenceLinearAlgebra.factor(a)
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv, singular = false)
        val ipiv = IntArray(n)
        val info = HostBlasCalls.dgetrf.invokeWithArguments(
            ROW_MAJOR,
            n,
            n,
            HostBlasCalls.seg(lu),
            n,
            HostBlasCalls.seg(ipiv),
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
     * Delegated to [ReferenceLinearAlgebra] for the same reason as [Blas.gemv]: two triangular solves over
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

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (n == 0 || nrhs == 0) return out
        // One or two columns do not cover the native call's cost: at n 256 dtrsm needs 100-140 us for a
        // single column against the delegated path's 25 us. From four columns on, dtrsm is ahead.
        if (nrhs < NATIVE_TRSM_MIN_RHS) {
            val col = workspace?.take(n) ?: DoubleArray(n)
            val solved = workspace?.take(n) ?: DoubleArray(n)
            for (c in 0 until nrhs) {
                for (i in 0 until n) col[i] = b.data[i * nrhs + c]
                solveInto(lu, col, solved, transpose, workspace)
                for (i in 0 until n) out.data[i * nrhs + c] = solved[i]
            }
            if (workspace != null) {
                workspace.release(solved)
                workspace.release(col)
            }
            return out
        }
        val f = lu.lu
        // Permute plus two block triangular solves; dtrsm is row-major native, so no LAPACKE
        // transposition tax scales with nrhs.
        if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmLeft(f, n, y, nrhs, UPPER, TRANS, NON_UNIT)
            trsmLeft(f, n, y, nrhs, LOWER, TRANS, UNIT)
            for (i in 0 until n) y.copyInto(out.data, lu.piv[i] * nrhs, i * nrhs, (i + 1) * nrhs)
            workspace?.release(y)
        } else {
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                for (i in 0 until n) b.data.copyInto(staged, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                for (i in 0 until n) {
                    b.data.copyInto(out.data, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
                }
            }
            trsmLeft(f, n, out.data, nrhs, LOWER, NO_TRANS, UNIT)
            trsmLeft(f, n, out.data, nrhs, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

    /** Left-side dtrsm over a packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsmLeft(f: DoubleArray, n: Int, x: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        HostBlasCalls.dtrsm.invokeWithArguments(
            ROW_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0,
            HostBlasCalls.seg(f), n, HostBlasCalls.seg(x), nrhs,
        )
    }

    override fun solveInto(
        ldl: LdlDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        workspace: Workspace?,
    ): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        val x = out.data
        if (out !== b) b.data.copyInto(x)
        if (n == 0 || nrhs == 0) return out
        val info = HostBlasCalls.dsytrs.invokeWithArguments(
            ROW_MAJOR, 'L'.code.toByte(), n, nrhs,
            HostBlasCalls.seg(ldl.ldl), n, HostBlasCalls.seg(ldl.ipiv), HostBlasCalls.seg(x), nrhs,
        ) as Int
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return out
    }

    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        if (a.rows < dispatchThresholds.lapack) return ReferenceLinearAlgebra.ldl(a, workspace)
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv, singular = false)
        val info = HostBlasCalls.dsytrf.invokeWithArguments(
            ROW_MAJOR,
            'L'.code.toByte(),
            n,
            HostBlasCalls.seg(buf),
            n,
            HostBlasCalls.seg(ipiv),
        ) as Int
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, singular = info > 0)
    }

    /**
     * Delegated to [ReferenceLinearAlgebra], like [Blas.gemv] and the LU vector solve: `dsytrs` on one
     * right-hand side is `O(n²)` work over the `O(n²)` factor, so the per-call cost cannot be amortized.
     * Measured at `n` 64 the portable path takes 3 us against 6, at 256 it is 51 against 141, and by 1024
     * the two converge as both become memory-bound — so delegating wins below and costs nothing above.
     * The blocked [solveInto] below stays native, where many right-hand sides amortize the same cost.
     */
    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.solveInto(ldl, b, out)

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        if (minOf(a.rows, a.cols) < dispatchThresholds.lapack) return ReferenceLinearAlgebra.qr(a, workspace)
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = HostBlasCalls.dgeqrf.invokeWithArguments(
                ROW_MAJOR,
                m,
                n,
                HostBlasCalls.seg(buf),
                n,
                HostBlasCalls.seg(tau),
            ) as Int
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    override fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        require(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        require(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
        val c = out
        if (out !== y) y.copyInto(out)
        if (qr.tau.isEmpty()) return c
        val side = 'L'.code.toByte()
        val trans = (if (transpose) 'T' else 'N').code.toByte()
        val info = HostBlasCalls.dormqr.invokeWithArguments(
            ROW_MAJOR, side, trans, qr.m, 1, qr.tau.size,
            HostBlasCalls.seg(qr.qr), qr.n, HostBlasCalls.seg(qr.tau), HostBlasCalls.seg(c), 1,
        ) as Int
        check(info == 0) { "dormqr: illegal argument ${-info}" }
        return c
    }

    override fun rcond(lu: LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        if (lu.n < dispatchThresholds.lapack) return ReferenceLinearAlgebra.rcond(lu, anorm, workspace)
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val out = DoubleArray(1)
        val info = HostBlasCalls.dgecon.invokeWithArguments(
            ROW_MAJOR,
            '1'.code.toByte(),
            n,
            HostBlasCalls.seg(lu.lu),
            n,
            anorm,
            HostBlasCalls.seg(out),
        ) as Int
        check(info == 0) { "dgecon: illegal argument ${-info}" }
        return out[0]
    }
}

private const val NATIVE_TRSM_MIN_RHS = 4
