package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.CholeskyDecomposition
import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.PivotedQrDecomposition
import com.eignex.koblas.dense.QrDecomposition
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.rankOfPivotedR
import com.eignex.koblas.dispatchThresholds
import com.eignex.koblas.hostblas.HostBlasCalls.COL_MAJOR
import com.eignex.koblas.hostblas.HostBlasCalls.LEFT
import com.eignex.koblas.hostblas.HostBlasCalls.LOWER
import com.eignex.koblas.hostblas.HostBlasCalls.NON_UNIT
import com.eignex.koblas.hostblas.HostBlasCalls.NO_TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.TRANS
import com.eignex.koblas.hostblas.HostBlasCalls.UNIT
import com.eignex.koblas.hostblas.HostBlasCalls.UPPER
import com.eignex.koblas.lapackFailedAt
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape

/** Dense factorizations backed by a host LAPACKE. */
public class HostLapack internal constructor() : Lapack {
    override val name: String get() = "openblas"

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override fun factor(a: DenseMatrix): LuDecomposition {
        if (a.rows < dispatchThresholds.lapack) return ReferenceLinearAlgebra.factor(a)
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv)
        val ipiv = IntArray(n)
        val info = HostBlasCalls.dgetrf.invokeWithArguments(
            COL_MAJOR,
            n,
            n,
            HostBlasCalls.seg(lu),
            n,
            HostBlasCalls.seg(ipiv),
        ) as Int
        check(info >= 0) { "dgetrf: illegal argument ${-info}" }
        // dgetrf reports successive 1-based row swaps, so replaying them gives the permutation form
        // LuDecomposition uses, where `piv(k)` is the original row now at position k.
        for (k in 0 until n) {
            val p = ipiv[k] - 1
            if (p != k) {
                val t = piv[k]
                piv[k] = piv[p]
                piv[p] = t
            }
        }
        return LuDecomposition(n, lu, piv, lapackFailedAt(info))
    }

    /** Delegated to [ReferenceLinearAlgebra] for the same reason as [Blas.gemv], the per-call cost. */
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
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        val nrhs = b.cols
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        requireShape(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (n == 0 || nrhs == 0) return out
        // Few columns do not cover the native call's cost.
        if (nrhs < NATIVE_TRSM_MIN_RHS) {
            val col = workspace?.take(n) ?: DoubleArray(n)
            val solved = workspace?.take(n) ?: DoubleArray(n)
            for (c in 0 until nrhs) {
                b.data.copyInto(col, 0, c * n, (c + 1) * n)
                solveInto(lu, col, solved, transpose, workspace)
                solved.copyInto(out.data, c * n, 0, n)
            }
            if (workspace != null) {
                workspace.release(solved)
                workspace.release(col)
            }
            return out
        }
        val f = lu.lu
        // Permute plus two block triangular solves.
        if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmLeft(f, n, y, nrhs, UPPER, TRANS, NON_UNIT)
            trsmLeft(f, n, y, nrhs, LOWER, TRANS, UNIT)
            permuteRows(y, out.data, n, nrhs, lu.piv, gather = false)
            workspace?.release(y)
        } else {
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                permuteRows(b.data, staged, n, nrhs, lu.piv, gather = true)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                permuteRows(b.data, out.data, n, nrhs, lu.piv, gather = true)
            }
            trsmLeft(f, n, out.data, nrhs, LOWER, NO_TRANS, UNIT)
            trsmLeft(f, n, out.data, nrhs, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

    /** `dst(i, c) = src(piv(i), c)` when [gather], its inverse otherwise, over an `n × nrhs` pair. */
    @Suppress("LongParameterList")
    private fun permuteRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray, gather: Boolean) {
        for (c in 0 until nrhs) {
            val base = c * n
            if (gather) {
                for (i in 0 until n) dst[base + i] = src[base + piv[i]]
            } else {
                for (i in 0 until n) dst[base + piv[i]] = src[base + i]
            }
        }
    }

    /** Left-side dtrsm over a packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsmLeft(f: DoubleArray, n: Int, x: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        HostBlasCalls.dtrsm.invokeWithArguments(
            COL_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0,
            HostBlasCalls.seg(f), n, HostBlasCalls.seg(x), n,
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
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        requireShape(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        val x = out.data
        if (out !== b) b.data.copyInto(x)
        if (n == 0 || nrhs == 0) return out
        val info = HostBlasCalls.dsytrs.invokeWithArguments(
            COL_MAJOR, 'L'.code.toByte(), n, nrhs,
            HostBlasCalls.seg(ldl.ldl), n, HostBlasCalls.seg(ldl.ipiv), HostBlasCalls.seg(x), n,
        ) as Int
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return out
    }

    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        if (a.rows < dispatchThresholds.lapack) return ReferenceLinearAlgebra.ldl(a, workspace)
        requireShape(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv)
        val info = HostBlasCalls.dsytrf.invokeWithArguments(
            COL_MAJOR,
            'L'.code.toByte(),
            n,
            HostBlasCalls.seg(buf),
            n,
            HostBlasCalls.seg(ipiv),
        ) as Int
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, lapackFailedAt(info))
    }

    /** Gated by [JVM_LDL_SOLVE_MIN] rather than by the LAPACK threshold, and disabled by it. */
    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        requireFactored(ldl.failedAt, "solve")
        if (ldl.n < JVM_LDL_SOLVE_MIN) return ReferenceLinearAlgebra.solveInto(ldl, b, out)
        val n = ldl.n
        requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
        requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
        if (out !== b) b.copyInto(out)
        if (n == 0) return out
        val info = HostBlasCalls.dsytrs.invokeWithArguments(
            COL_MAJOR, 'L'.code.toByte(), n, 1,
            HostBlasCalls.seg(ldl.ldl), n, HostBlasCalls.seg(ldl.ipiv), HostBlasCalls.seg(out), n,
        ) as Int
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return out
    }

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        if (minOf(a.rows, a.cols) < dispatchThresholds.lapack) return ReferenceLinearAlgebra.qr(a, workspace)
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = HostBlasCalls.dgeqrf.invokeWithArguments(
                COL_MAJOR,
                m,
                n,
                HostBlasCalls.seg(buf),
                m,
                HostBlasCalls.seg(tau),
            ) as Int
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    /**
     * A zeroed jpvt on input leaves every column free to move, and the output is one-based, hence the shift.
     * LAPACK reports no rank, so it comes from [rankOfPivotedR].
     */
    override fun qrPivoted(a: DenseMatrix, tolerance: Double, workspace: Workspace?): PivotedQrDecomposition {
        if (minOf(a.rows, a.cols) < dispatchThresholds.lapack) {
            return ReferenceLinearAlgebra.qrPivoted(a, tolerance, workspace)
        }
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        val jpvt = IntArray(n) // zero: no column is pinned
        if (m > 0 && n > 0) {
            val info = HostBlasCalls.dgeqp3.invokeWithArguments(
                COL_MAJOR,
                m,
                n,
                HostBlasCalls.seg(buf),
                m,
                HostBlasCalls.seg(jpvt),
                HostBlasCalls.seg(tau),
            ) as Int
            check(info == 0) { "dgeqp3: illegal argument ${-info}" }
        }
        val pivots = IntArray(n) { jpvt[it] - 1 }
        val rank = rankOfPivotedR(buf, m, n, minOf(m, n), tolerance)
        return PivotedQrDecomposition(QrDecomposition(m, n, buf, tau), pivots, rank)
    }

    override fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        requireShape(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        requireShape(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
        val c = out
        if (out !== y) y.copyInto(out)
        if (qr.tau.isEmpty()) return c
        val side = 'L'.code.toByte()
        val trans = (if (transpose) 'T' else 'N').code.toByte()
        val info = HostBlasCalls.dormqr.invokeWithArguments(
            COL_MAJOR, side, trans, qr.m, 1, qr.tau.size,
            HostBlasCalls.seg(qr.qr), qr.m, HostBlasCalls.seg(qr.tau), HostBlasCalls.seg(c), qr.m,
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
            COL_MAJOR,
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

    /**
     * Clears the upper triangle LAPACK leaves untouched, reads only the lower triangle of the input, and
     * falls back to the portable path on a positive info, which [CholeskyPolicy.Regularize] needs.
     */
    override fun cholesky(a: DenseMatrix, policy: CholeskyPolicy): CholeskyDecomposition {
        if (a.rows < JVM_CHOLESKY_MIN) return super.cholesky(a, policy)
        requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (n == 0) return CholeskyDecomposition(DenseMatrix(0, 0))
        val l = DenseMatrix(n, n)
        // One bulk copy per column of the lower triangle; dpotrf reads no further.
        for (j in 0 until n) a.data.copyInto(l.data, j + j * n, j + j * n, (j + 1) * n)
        val info = HostBlasCalls.dpotrf.invokeWithArguments(
            COL_MAJOR,
            'L'.code.toByte(),
            n,
            HostBlasCalls.seg(l.data),
            n,
        ) as Int
        check(info >= 0) { "dpotrf: illegal argument ${-info}" }
        if (info > 0) return super.cholesky(a, policy)
        for (i in 0 until n) for (j in i + 1 until n) l[i, j] = 0.0
        return CholeskyDecomposition(l)
    }

    /** Gated by [JVM_SOLVE_SPD_MIN], which shuts it off. */
    override fun solve(chol: CholeskyDecomposition, b: DoubleArray): DoubleArray {
        if (chol.n < JVM_SOLVE_SPD_MIN) return super.solve(chol, b)
        val n = chol.n
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        val x = b.copyOf()
        if (n == 0) return x
        val info = HostBlasCalls.dpotrs.invokeWithArguments(
            COL_MAJOR,
            'L'.code.toByte(),
            n,
            1,
            HostBlasCalls.seg(chol.l.data),
            n,
            HostBlasCalls.seg(x),
            n,
        ) as Int
        check(info == 0) { "dpotrs: illegal argument ${-info}" }
        return x
    }

    /**
     * dpotri writes only the triangle it is given, so the result is mirrored, and it overwrites the factor,
     * so the factor is copied first.
     */
    override fun invert(chol: CholeskyDecomposition, workspace: Workspace?): DenseMatrix {
        if (chol.n < JVM_INVERT_SPD_MIN) return super.invert(chol, workspace)
        val n = chol.n
        if (n == 0) return DenseMatrix(0, 0)
        val inv = DenseMatrix(n, n, chol.l.data.copyOf())
        val info = HostBlasCalls.dpotri.invokeWithArguments(
            COL_MAJOR,
            'L'.code.toByte(),
            n,
            HostBlasCalls.seg(inv.data),
            n,
        ) as Int
        check(info >= 0) { "dpotri: illegal argument ${-info}" }
        check(info == 0) { "dpotri: zero diagonal at $info, the factor is singular" }
        for (i in 0 until n) for (j in 0 until i) inv[j, i] = inv[i, j]
        return inv
    }
}

private const val NATIVE_TRSM_MIN_RHS = 4

/** Holds the LDL vector solve on the portable path at every size, see [HostLapack.solveInto]. */
private const val JVM_LDL_SOLVE_MIN = 1 shl 20

/** Where dpotrf takes over from the portable Cholesky. */
private const val JVM_CHOLESKY_MIN = 32

/** Where dpotri takes over from the portable SPD inverse. */
private const val JVM_INVERT_SPD_MIN = 16

/** Holds the SPD solve on the portable path at every size, see [HostLapack.solve]. */
private const val JVM_SOLVE_SPD_MIN = 1 shl 20
