@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L

package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.cblas.Cblas.LEFT
import com.eignex.koblas.dense.host.cblas.Cblas.LOWER
import com.eignex.koblas.dense.host.cblas.Cblas.NON_UNIT
import com.eignex.koblas.dense.host.cblas.Cblas.NO_TRANS
import com.eignex.koblas.dense.host.cblas.Cblas.TRANS
import com.eignex.koblas.dense.host.cblas.Cblas.UNIT
import com.eignex.koblas.dense.host.cblas.Cblas.UPPER
import com.eignex.koblas.internal.backend.DispatchThresholds

/** LAPACK's lower-triangle selector, which koblas asks for everywhere it has a choice. */
private const val LOWER_UPLO: Byte = 'L'.code.toByte()

/** LAPACK's left-side selector. The same byte as [LOWER_UPLO], but a different argument. */
private const val SIDE_LEFT: Byte = 'L'.code.toByte()

/**
 * The dense factorizations a host LAPACKE provides, over whichever [LapackeCalls] the platform supplies.
 * Both host bindings are this class plus their own FFI mechanism.
 *
 * The size at which a native call starts to pay differs by host, so every gate arrives in [dispatch],
 * resolved from the binding's own configuration. Its defaults dispatch natively at any size.
 */
@Suppress("TooManyFunctions") // the LAPACK surface a host library covers
public abstract class F64DecompositionsAdapter internal constructor(
    private val f: LapackeCalls,
    private val blas: CblasCalls,
    private val portable: F64ReferenceDecompositions = F64ReferenceDecompositions(),
    private val dispatch: DispatchThresholds = DispatchThresholds(0, 0, 0, 0),
) : F64Decompositions {

    /** A binding that calls out, whatever the portable instance it falls back to reports. */
    override val isPortable: Boolean get() = false

    // No host binding for these, so they run the portable versions. Forwarded explicitly rather than by
    // class delegation, which would route a caller's convenience overloads to the portable routine instead
    // of the accelerated one, since a delegated member calls back into the delegate.
    override fun invert(lu: F64LuDecomposition, workspace: Workspace?): F64DenseMatrix = portable.invert(lu, workspace)

    override fun trtri(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean): F64DenseMatrix =
        portable.trtri(a, lower, unitDiag)

    override fun solveInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        minimumNorm: Boolean,
        workspace: Workspace?,
    ): DoubleArray = portable.solveInto(qr, b, out, minimumNorm, workspace)

    override fun solveInto(
        qr: F64PivotedQrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray = portable.solveInto(qr, b, out, workspace)

    /** Performs `dgeqp3`, or returns null when the binding does not expose it. */
    protected open fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? = null

    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition {
        requireRankTolerance(tolerance)
        if (minOf(a.rows, a.cols) < maxOf(1, dispatch.factorize)) {
            return portable.qrPivoted(a, tolerance, workspace)
        }
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        val jpvt = IntArray(n)
        val info = dgeqp3(m, n, buf, jpvt, tau) ?: return portable.qrPivoted(a, tolerance, workspace)
        check(info == 0) { "dgeqp3: illegal argument ${-info}" }
        val pivots = IntArray(n) { jpvt[it] - 1 }
        val rank = rankOfPivotedR(buf, m, n, minOf(m, n), tolerance)
        return F64PivotedQrDecomposition(F64QrDecomposition(m, n, buf, tau), pivots, rank)
    }

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        if (a.rows < dispatch.factorize) return portable.factor(a)
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return factorInto(a, F64LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    /** `dgetrf` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        if (a.rows < dispatch.factorize) return portable.factorInto(a, out)
        val n = out.n
        a.data.copyInto(out.lu)
        val piv = out.piv
        for (i in 0 until n) piv[i] = i
        out.failedAt = NOT_SINGULAR
        if (n == 0) return out
        val ipiv = IntArray(n)
        val info = f.dgetrf(COL_MAJOR, n, n, out.lu, n, ipiv)
        check(info >= 0) { "dgetrf: illegal argument ${-info}" }
        // dgetrf reports successive 1-based row swaps, so replaying them gives the permutation form
        // F64LuDecomposition uses, where `piv(k)` is the original row now at position k.
        for (k in 0 until n) {
            val p = ipiv[k] - 1
            if (p != k) {
                val t = piv[k]
                piv[k] = piv[p]
                piv[p] = t
            }
        }
        out.failedAt = lapackFailedAt(info)
        return out
    }

    /** One right-hand side is a pair of triangular solves, a wider one a pair of `dtrsm`. */
    private fun dispatchesSolve(n: Int, nrhs: Int): Boolean = if (nrhs <= 1) {
        n >= dispatch.level2
    } else {
        n.toLong() * nrhs >= dispatch.level3.toLong() * dispatch.level3
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        if (!dispatchesSolve(n, 1)) return portable.solveInto(lu, b, out, transpose, workspace)
        requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
        requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
        if (n == 0) return out
        nativeSolve(lu, F64DenseMatrix.wrap(n, 1, b), F64DenseMatrix.wrap(n, 1, out), transpose, workspace)
        return out
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        if (n == 0 || nrhs == 0) return out
        if (!dispatchesSolve(n, nrhs)) return portable.solveInto(lu, b, out, transpose, workspace)
        return nativeSolve(lu, b, out, transpose, workspace)
    }

    /** The `dgetrs` body both solve overloads share, once the gate has chosen to cross. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    private fun nativeSolve(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        val factor = lu.lu
        if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            // Handed back even when a native solve raises, for the reason the syrk borrow gives.
            try {
                b.data.copyInto(y)
                trsmLeft(factor, n, y, nrhs, UPPER, TRANS, NON_UNIT)
                trsmLeft(factor, n, y, nrhs, LOWER, TRANS, UNIT)
                permuteRows(y, out.data, n, nrhs, lu.piv, gather = false)
            } finally {
                workspace?.release(y)
            }
        } else {
            if (out.data === b.data) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                try {
                    permuteRows(b.data, staged, n, nrhs, lu.piv, gather = true)
                    staged.copyInto(out.data)
                } finally {
                    workspace?.release(staged)
                }
            } else {
                permuteRows(b.data, out.data, n, nrhs, lu.piv, gather = true)
            }
            trsmLeft(factor, n, out.data, nrhs, LOWER, NO_TRANS, UNIT)
            trsmLeft(factor, n, out.data, nrhs, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

    /** Left-side dtrsm over a packed factor buffer. */
    @Suppress("LongParameterList") // the dtrsm selectors the caller varies
    private fun trsmLeft(factor: DoubleArray, n: Int, x: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        blas.dtrsm(COL_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0, factor, n, x, n)
    }

    override fun ldl(a: F64DenseMatrix, workspace: Workspace?): F64LdlDecomposition {
        if (a.rows < dispatch.factorize) return portable.ldl(a, workspace)
        requireShape(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return F64LdlDecomposition(0, buf, ipiv)
        val info = f.dsytrf(COL_MAJOR, LOWER_UPLO, n, buf, n, ipiv)
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return F64LdlDecomposition(n, buf, ipiv, lapackFailedAt(info))
    }

    /** The vector solve stays portable on both bindings: one `dsytrs` call does not cover its own cost. */
    override fun solveInto(ldl: F64LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        portable.solveInto(ldl, b, out)

    /** Native only from the configured right-hand-side count, as for the LU multi-RHS solve above. */
    override fun solveInto(
        ldl: F64LdlDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(ldl.failedAt, "solve")
        if (!dispatchesSolve(b.rows, b.cols)) return portable.solveInto(ldl, b, out, workspace)
        val n = ldl.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        val x = out.data
        if (out !== b) b.data.copyInto(x)
        if (n == 0 || nrhs == 0) return out
        val info = f.dsytrs(COL_MAJOR, LOWER_UPLO, n, nrhs, ldl.ldl, n, ldl.ipiv, x, n)
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return out
    }

    override fun qr(a: F64DenseMatrix, workspace: Workspace?): F64QrDecomposition {
        if (minOf(a.rows, a.cols) < dispatch.factorize) return portable.qr(a, workspace)
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = f.dgeqrf(COL_MAJOR, m, n, buf, m, tau)
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return F64QrDecomposition(m, n, buf, tau)
    }

    override fun applyQInto(qr: F64QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        // Gated on the work, m reflector applications of length k, against what the square case costs at
        // the threshold. Gating on min(m, n) alone would keep a tall thin factorization portable however
        // large m grows, which is the shape least-squares presents most often.
        val reflectorWork = qr.m.toLong() * qr.tau.size
        val gate = dispatch.factorize.toLong() * dispatch.factorize
        if (reflectorWork < gate) return portable.applyQInto(qr, y, out, transpose)
        requireShape(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        requireShape(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
        if (out !== y) y.copyInto(out)
        if (qr.tau.isEmpty()) return out
        val trans = (if (transpose) 'T' else 'N').code.toByte()
        val info = f.dormqr(
            COL_MAJOR, SIDE_LEFT, trans, qr.m, 1, qr.tau.size, qr.qr, qr.m, qr.tau,
            out, qr.m,
        )
        check(info == 0) { "dormqr: illegal argument ${-info}" }
        return out
    }

    override fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        if (lu.n < dispatch.factorize) return portable.rcond(lu, anorm, workspace)
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val out = DoubleArray(1)
        val info = f.dgecon(COL_MAJOR, '1'.code.toByte(), n, lu.lu, n, anorm, out)
        check(info == 0) { "dgecon: illegal argument ${-info}" }
        return out[0]
    }

    /**
     * Reads only the lower triangle of the input, and falls back to the portable path on a positive info,
     * which [CholeskyPolicy.Regularize] needs.
     */
    override fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy): F64CholeskyDecomposition {
        if (a.rows < dispatch.factorize) return portable.cholesky(a, policy)
        requireSquare(a, "cholesky")
        val n = a.rows
        if (n == 0) return F64CholeskyDecomposition(F64DenseMatrix(0, 0))
        val l = F64DenseMatrix(n, n)
        // One bulk copy per column of the lower triangle; dpotrf reads no further.
        for (j in 0 until n) a.data.copyInto(l.data, j + j * n, j + j * n, (j + 1) * n)
        val info = f.dpotrf(COL_MAJOR, LOWER_UPLO, n, l.data, n)
        check(info >= 0) { "dpotrf: illegal argument ${-info}" }
        if (info > 0) return portable.cholesky(a, policy)
        // The destination is allocated zeroed and `dpotrf` with `uplo = L` leaves the strict upper triangle
        // untouched, so it is already the zero this returns.
        return F64CholeskyDecomposition(l)
    }

    /**
     * dpotri writes only the triangle it is given, so the result is mirrored, and it overwrites the factor,
     * so the factor is copied first.
     */
    override fun invert(chol: F64CholeskyDecomposition, workspace: Workspace?): F64DenseMatrix {
        if (chol.n < dispatch.factorize) return portable.invert(chol, workspace)
        val n = chol.n
        if (n == 0) return F64DenseMatrix(0, 0)
        val inv = F64DenseMatrix(n, n, chol.l.data.copyOf())
        val info = f.dpotri(COL_MAJOR, LOWER_UPLO, n, inv.data, n)
        check(info >= 0) { "dpotri: illegal argument ${-info}" }
        // A zero on the diagonal is dpotri's error and the reference's division by zero, and the factor is
        // the caller's to construct over any square matrix. The reference answers with infinities, so this
        // half answers the same way rather than throwing where it would not.
        if (info > 0) return portable.invert(chol, workspace)
        // Column i's upper entries are contiguous, so the mirror writes down a column and reads across.
        val invd = inv.data
        for (i in 0 until n) {
            val base = inv.colOffset(i)
            for (j in 0 until i) invd[base + j] = invd[i + j * n]
        }
        return inv
    }
}
