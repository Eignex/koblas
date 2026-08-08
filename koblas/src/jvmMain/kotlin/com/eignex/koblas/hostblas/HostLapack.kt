package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.MatrixLike
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.PivotedQrDecomposition
import com.eignex.koblas.dense.QrDecomposition
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.invertSpd
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.rankOfPivotedR
import com.eignex.koblas.dense.solveSpd
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

/**
 * The host LAPACKE as the JVM's [Lapack] half.
 *
 * Native where the work is `O(n^3)`: the LU, LDL and QR factorizations, the blocked multi-RHS solve, and
 * the condition estimate. The single-vector solves delegate to the portable kernels, as on every other
 * platform.
 *
 * The Cholesky family is native from a small size up, and the solve is not: `dpotrf` and `dpotri` win from
 * n 32 and n 16 respectively, while `dpotrs` loses two to three times at every size. That split is the whole
 * argument for per-routine gates — the three routines of one family disagree about where the call is worth
 * making, and only measuring finds out.
 */
class HostLapack internal constructor() : Lapack {
    override val name: String get() = "openblas"

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override fun factor(a: DenseMatrix): LuDecomposition {
        if (a.rows < dispatchThresholds.lapack) return ReferenceLinearAlgebra.factor(a)
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
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
        return LuDecomposition(n, lu, piv, lapackFailedAt(info))
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
        // Permute plus two block triangular solves. The row permutation is the strided step now: it moves
        // one element per column instead of a contiguous run per row.
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

    /** `dst[i, c] = src[piv[i], c]` when [gather], its inverse otherwise, over an `n × nrhs` pair. */
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
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
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
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
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

    /**
     * Gated by [JVM_LDL_SOLVE_MIN] rather than by the LAPACK threshold, and disabled by it.
     *
     * A vector solve is `O(n^2)` work over `O(n^2)` data, so it behaves like level 2 and not like the
     * factorization it belongs to. Measured against SIMD it stayed inside the noise band at every size —
     * 1.10x, 1.02x, 1.08x, 1.02x at n=16..128, and SIMD ahead 1.08x at 256 — so there is nothing to win.
     * The gate exists so that is a value someone can change and re-measure, not a decision welded shut.
     */
    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        if (ldl.n < JVM_LDL_SOLVE_MIN) return ReferenceLinearAlgebra.solveInto(ldl, b, out)
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
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
     * `dgeqp3` above the LAPACK threshold, portable below it.
     *
     * `jpvt` is zeroed on input, which is LAPACK's "every column is free to move"; a nonzero entry pins a
     * column to the front, which koblas does not expose. On output it is one-based, hence the shift.
     *
     * LAPACK reports no rank, so it comes from [rankOfPivotedR] — the same rule the portable path applies to
     * its own `R`, rather than a second implementation that could disagree about where the tolerance falls.
     *
     * Whether this is worth the call is not settled. `dgeqp3`'s pivoting phase updates column norms between
     * panels and is level-2 bound, so the level-3 advantage that makes `dgeqrf` worth binding is smaller
     * here; this reuses the LAPACK threshold rather than claiming a measured one of its own.
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
        require(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        require(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
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
     * `dpotrf` above [JVM_CHOLESKY_MIN], portable below it, with three adjustments to match the contract.
     *
     * LAPACK factorizes in place and leaves the strict upper triangle exactly as the input had it, while
     * koblas returns a factor whose upper triangle is zero, so it is cleared. LAPACK has no equivalent of
     * [CholeskyPolicy.Regularize]: it reports the failing leading minor instead of clamping the pivot, so `info > 0`
     * falls back to the portable path, which is what applies the clamp or throws per the flag. And only the
     * lower triangle of the input is read, matching what koblas promises its callers.
     *
     * The threshold is where the measurement puts it, not at the LAPACK default: SIMD won 2.19x at n=256,
     * tied at 1024, and lost 1.95x at 2048, so this gate opens late.
     */
    override fun cholesky(a: MatrixLike, policy: CholeskyPolicy): DenseMatrix {
        if (a.rows < JVM_CHOLESKY_MIN) return super.cholesky(a, policy)
        require(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (n == 0) return DenseMatrix(0, 0)
        val l = DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0..i) l[i, j] = a[i, j]
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
        return l
    }

    /**
     * Gated by [JVM_SOLVE_SPD_MIN], which shuts it: `dpotrs` lost to the SIMD kernels by 3x at n=256 and
     * 12x at 2048. It is `O(n^2)` work over `O(n^2)` data, so there is nothing to amortize a call against.
     *
     * Those numbers were taken under row-major storage, when a second cause compounded the first:
     * LAPACKE transposed the matrix into a column-major temporary on every call. That cost is gone now
     * that koblas stores what LAPACK wants, so the margin here is narrower than measured and the gate is
     * a stale upper bound rather than a current answer; see the note above the constants below.
     */
    override fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray {
        if (L.rows < JVM_SOLVE_SPD_MIN) return super.solveSpd(L, b)
        val n = L.rows
        require(b.size == n) { "solveSpd: b size ${b.size}, expected $n" }
        val x = b.copyOf()
        if (n == 0) return x
        val info = HostBlasCalls.dpotrs.invokeWithArguments(
            COL_MAJOR,
            'L'.code.toByte(),
            n,
            1,
            HostBlasCalls.seg(L.data),
            n,
            HostBlasCalls.seg(x),
            n,
        ) as Int
        check(info == 0) { "dpotrs: illegal argument ${-info}" }
        return x
    }

    /**
     * `dpotri` above [JVM_INVERT_SPD_MIN]. It writes only the triangle it is given, where koblas returns the
     * full symmetric inverse, so the result is mirrored; and it overwrites the factor with the inverse, so
     * the factor is copied first rather than destroyed.
     */
    override fun invertSpd(L: DenseMatrix, workspace: Workspace?): DenseMatrix {
        if (L.rows < JVM_INVERT_SPD_MIN) return super.invertSpd(L, workspace)
        val n = L.rows
        if (n == 0) return DenseMatrix(0, 0)
        val inv = DenseMatrix(n, n, L.data.copyOf())
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

// Every threshold below, and the level defaults in DispatchThresholds.jvm.kt, was measured while koblas
// stored matrices row-major.
//
// Two things changed underneath those numbers. The native side got cheaper: LAPACKE's row-major entry
// points used to transpose each matrix into a column-major temporary before calling LAPACK — 32 MB of
// copying per call at n=2048 — and now nothing is transposed. The portable side changed shape too, since
// the kernels were rewritten to sweep columns rather than rows. Both effects move the crossovers, and not
// necessarily in the same direction.
//
// So these are the last measured answers under the old layout, kept because a stale measurement beats a
// guess, and every one of them is a candidate to move. The gates exist precisely so that re-measuring
// changes a constant rather than a method.

/**
 * The LDL vector solve stays portable at every size: see [HostLapack.solveInto]. Not the never-dispatch sentinel but a
 * finite bound, because the claim rests on measurements that stop at n=256 rather than on a proof.
 */
private const val JVM_LDL_SOLVE_MIN = 1 shl 20

/**
 * Native from 32 up (us/op, native against SIMD): 0.83 vs 0.87 at n=16, 2.79 vs 3.56 at 32, 11.3 vs 15.5 at
 * 64, 73.5 vs 83.6 at 128, 392 vs 564 at 256, 18353 vs 33498 at 1024, 98988 vs 295342 at 2048.
 *
 * This gate used to sit at 2048 on the opposite conclusion — that the portable factorization won until then —
 * and that measurement was taken under row-major storage, when every LAPACKE call transposed the matrix into
 * a column-major temporary first. koblas stores what LAPACK wants now, so the transposes are gone and the
 * ordering reversed. Sixteen is a tie within the error bars and is left to the portable path; below that is
 * not measured, and a matrix that small is not worth a foreign call.
 */
private const val JVM_CHOLESKY_MIN = 32

/**
 * Native at every size measured, and by the widest margin of the three (us/op, native against SIMD): 1.52 vs
 * 2.74 at n=16, 5.79 vs 12.0 at 32, 25.6 vs 54.1 at 64, 126 vs 287 at 128, 746 vs 1632 at 256, 20930 vs 81472
 * at 1024, 132975 vs 674060 at 2048.
 *
 * Two-to-five times, consistently, where the old note called this the least certain of the three at close to
 * the noise floor — the same row-major transposes that moved the Cholesky gate. Sixteen because that is the
 * smallest size measured, not because a crossover was found there; there is no sign of one.
 */
private const val JVM_INVERT_SPD_MIN = 16

/**
 * The SPD solve stays portable at every size: see [HostLapack.solveSpd]. Re-measured under column-major
 * storage rather than inherited — 38.4 us against the portable 15.3 at n=256, 655 against 234 at 1024, 2937
 * against 1079 at 2048 — so this one keeps its sentinel for the reason it was given, not despite it. Two
 * triangular solves are `O(n²)` work over `O(n²)` data and there is nothing for a foreign call to amortize.
 */
private const val JVM_SOLVE_SPD_MIN = 1 shl 20
