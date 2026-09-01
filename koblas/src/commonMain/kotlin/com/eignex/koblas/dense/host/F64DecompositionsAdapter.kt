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

/** LAPACK's lower-triangle selector, which koblas asks for everywhere it has a choice. */
private const val LOWER_UPLO: Byte = 'L'.code.toByte()
private const val UPPER_UPLO: Byte = 'U'.code.toByte()
private const val UNIT_DIAG: Byte = 'U'.code.toByte()
private const val NON_UNIT_DIAG: Byte = 'N'.code.toByte()

/** LAPACK's left-side selector. The same byte as [LOWER_UPLO], but a different argument. */
private const val SIDE_LEFT: Byte = 'L'.code.toByte()

/**
 * The dense factorizations a host LAPACKE provides, over whichever [LapackeCalls] the platform supplies.
 * Both host bindings are this class plus their own FFI mechanism.
 */
@Suppress("TooManyFunctions") // the LAPACK surface a host library covers
public abstract class F64DecompositionsAdapter internal constructor(
    private val f: LapackeCalls,
    private val blas: CblasCalls,
    private val portable: F64ReferenceDecompositions = F64ReferenceDecompositions(),
    private val metadata: BackendMetadata = BackendMetadata(integerAbi = "LP64"),
) : F64Decompositions,
    F64RoutingBackend,
    BackendMetadataProvider {

    /** A binding that calls out, whatever the portable instance it falls back to reports. */
    override val isPortable: Boolean get() = false

    override val backendMetadata: BackendMetadata get() = metadata

    override fun route(query: F64RouteQuery): BackendRoute? = when (query) {
        is F64RouteQuery.DenseLu -> nativeRoute(query, this, fallbackWhenUnavailable = false)
        else -> null
    }

    // Forwarded explicitly rather than by class delegation, which would route a caller's convenience
    // overloads to the portable routine instead of the accelerated one, since a delegated member calls back
    // into the delegate.
    override fun invert(lu: F64LuDecomposition, workspace: Workspace?): F64DenseMatrix {
        requireLuSquare(lu, "invert")
        if (lu.singular) {
            throw SingularMatrix(
                lu.failedAt,
                "invert: factorization is singular at pivot ${lu.failedAt}, so the inverse does not exist",
            )
        }
        val n = lu.n
        if (n == 0) return F64DenseMatrix(0, 0)
        val inv = F64DenseMatrix(n, n, lu.lu.copyOf())
        val info = f.dgetri(COL_MAJOR, n, inv.data, n, lapackPivots(lu.mutablePivots, n))
        check(info >= 0) { "dgetri: illegal argument ${-info}" }
        // A zero pivot is dgetri's error and the reference's division by zero, and the factorization is the
        // caller's to hold. The reference answers with infinities, so this half answers the same way.
        if (info > 0) return portable.invert(lu, workspace)
        return inv
    }

    /**
     * [F64LuDecomposition] holds the permutation, `piv(k)` being the original row now at k, where LAPACK
     * wants the successive one-based swaps `dgetrf` reports. Replaying the permutation recovers a swap
     * sequence that produces it.
     */
    private fun lapackPivots(piv: IntArray, n: Int): IntArray {
        val row = IntArray(n) { it }
        val position = IntArray(n) { it }
        val ipiv = IntArray(n)
        for (k in 0 until n) {
            val j = position[piv[k]]
            ipiv[k] = j + 1
            val here = row[k]
            val there = row[j]
            row[k] = there
            row[j] = here
            position[there] = k
            position[here] = j
        }
        return ipiv
    }

    override fun trtri(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean): F64DenseMatrix {
        requireShape(a.rows == a.cols) { "trtri requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (n == 0) return F64DenseMatrix(0, 0)
        if (!unitDiag) {
            for (i in 0 until n) {
                if (a[i, i] == 0.0) {
                    throw SingularMatrix(i, "trtri: triangle is singular, diagonal entry $i is zero")
                }
            }
        }
        val inv = F64DenseMatrix(n, n, a.data.copyOf())
        val info = f.dtrtri(
            COL_MAJOR,
            if (lower) LOWER_UPLO else UPPER_UPLO,
            if (unitDiag) UNIT_DIAG else NON_UNIT_DIAG,
            n,
            inv.data,
            n,
        )
        check(info >= 0) { "dtrtri: illegal argument ${-info}" }
        if (info > 0) return portable.trtri(a, lower, unitDiag)
        // dtrtri leaves the untouched triangle as it found it and a unit diagonal unwritten, where the
        // reference returns the triangle alone over zeros with the diagonal filled in.
        val invd = inv.data
        for (j in 0 until n) {
            val base = j * n
            if (lower) {
                for (i in 0 until j) invd[base + i] = 0.0
            } else {
                for (i in j + 1 until n) invd[base + i] = 0.0
            }
            if (unitDiag) invd[base + j] = 1.0
        }
        return inv
    }

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
        if (a.rows == 0 || a.cols == 0) return portable.qrPivoted(a, tolerance, workspace)
        val m = a.rows
        val n = a.cols
        return qrPivotedInto(
            a,
            F64PivotedQrDecomposition(
                F64QrDecomposition(m, n, DoubleArray(a.data.size), DoubleArray(minOf(m, n))),
                IntArray(n),
                rank = 0,
            ),
            tolerance,
            workspace,
        )
    }

    /** `dgeqp3` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun qrPivotedInto(
        a: F64DenseMatrix,
        out: F64PivotedQrDecomposition,
        tolerance: Double,
        workspace: Workspace?,
    ): F64PivotedQrDecomposition {
        requireRankTolerance(tolerance)
        requireShape(out.m == a.rows && out.n == a.cols) {
            "qrPivotedInto: out is ${out.m}x${out.n}, expected ${a.rows}x${a.cols}"
        }
        val m = a.rows
        val n = a.cols
        if (m == 0 || n == 0) return portable.qrPivotedInto(a, out, tolerance, workspace)
        val buf = out.factorization.qr
        a.data.copyInto(buf)
        // dgeqp3 reads jpvt as the caller's leading-column request, so it starts at "no column is fixed".
        val jpvt = IntArray(n)
        val info = dgeqp3(m, n, buf, jpvt, out.factorization.tau)
            ?: return portable.qrPivotedInto(a, out, tolerance, workspace)
        check(info == 0) { "dgeqp3: illegal argument ${-info}" }
        for (c in 0 until n) out.pivots[c] = jpvt[c] - 1
        out.rank = rankOfPivotedR(buf, m, n, minOf(m, n), tolerance)
        return out
    }

    override fun factor(a: F64DenseMatrix): F64LuDecomposition =
        factorInto(a, F64LuDecomposition(a.rows, a.cols, DoubleArray(a.data.size), IntArray(a.rows)))

    /** `dgetrf` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(out.rows == a.rows && out.cols == a.cols) {
            "factorInto: out is ${out.rows}x${out.cols}, expected ${a.rows}x${a.cols}"
        }
        val m = out.rows
        val n = out.cols
        val order = out.order
        a.data.copyInto(out.lu)
        val piv = out.mutablePivots
        for (i in 0 until m) piv[i] = i
        out.failedAt = NOT_SINGULAR
        if (order == 0) return out
        val ipiv = IntArray(order)
        val info = f.dgetrf(COL_MAJOR, m, n, out.lu, m, ipiv)
        check(info >= 0) { "dgetrf: illegal argument ${-info}" }
        // dgetrf reports successive 1-based row swaps, so replaying them gives the permutation form
        // F64LuDecomposition uses, where `piv(k)` is the original row now at position k.
        for (k in 0 until order) {
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

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        requireLuSquare(lu, "solve")
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
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
        requireLuSquare(lu, "solve")
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        if (n == 0 || nrhs == 0) return out
        return nativeSolve(lu, b, out, transpose, workspace)
    }

    /** The `dgetrs` body both solve overloads share. */
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
                permuteRows(y, out.data, n, nrhs, lu.mutablePivots, gather = false)
            } finally {
                workspace?.release(y)
            }
        } else {
            if (out.data === b.data) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                try {
                    permuteRows(b.data, staged, n, nrhs, lu.mutablePivots, gather = true)
                    staged.copyInto(out.data)
                } finally {
                    workspace?.release(staged)
                }
            } else {
                permuteRows(b.data, out.data, n, nrhs, lu.mutablePivots, gather = true)
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

    override fun pivotedSymmetricIndefinite(
        a: F64DenseMatrix,
        workspace: Workspace?,
    ): F64PivotedSymmetricIndefiniteDecomposition {
        requireShape(a.rows == a.cols) {
            "pivotedSymmetricIndefinite: matrix must be square, got ${a.rows}x${a.cols}"
        }
        val n = a.rows
        return pivotedSymmetricIndefiniteInto(
            a,
            F64PivotedSymmetricIndefiniteDecomposition(n, DoubleArray(n * n), IntArray(n)),
            workspace,
        )
    }

    /** `dsytrf` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun pivotedSymmetricIndefiniteInto(
        a: F64DenseMatrix,
        out: F64PivotedSymmetricIndefiniteDecomposition,
        workspace: Workspace?,
    ): F64PivotedSymmetricIndefiniteDecomposition {
        requireShape(a.rows == a.cols) {
            "pivotedSymmetricIndefinite: matrix must be square, got ${a.rows}x${a.cols}"
        }
        val n = a.rows
        requireShape(out.n == n) {
            "pivotedSymmetricIndefiniteInto: out is ${out.n}x${out.n}, expected ${n}x$n"
        }
        a.data.copyInto(out.ldl)
        out.failedAt = NOT_SINGULAR
        if (n == 0) return out
        val info = f.dsytrf(COL_MAJOR, LOWER_UPLO, n, out.ldl, n, out.rawLapackIpiv)
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        out.failedAt = lapackFailedAt(info)
        return out
    }

    /** The vector solve stays portable on both bindings: one `dsytrs` call does not cover its own cost. */
    override fun solveInto(
        factor: F64PivotedSymmetricIndefiniteDecomposition,
        b: DoubleArray,
        out: DoubleArray,
    ): DoubleArray = portable.solveInto(factor, b, out)

    /** Solves several right-hand sides with the host factorization. */
    override fun solveInto(
        factor: F64PivotedSymmetricIndefiniteDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(factor.failedAt, "solve")
        val n = factor.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        val x = out.data
        if (out !== b) b.data.copyInto(x)
        if (n == 0 || nrhs == 0) return out
        val info = f.dsytrs(COL_MAJOR, LOWER_UPLO, n, nrhs, factor.ldl, n, factor.rawLapackIpiv, x, n)
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return out
    }

    override fun qr(a: F64DenseMatrix, workspace: Workspace?): F64QrDecomposition = qrInto(
        a,
        F64QrDecomposition(a.rows, a.cols, DoubleArray(a.data.size), DoubleArray(minOf(a.rows, a.cols))),
        workspace,
    )

    /** `dgeqrf` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun qrInto(a: F64DenseMatrix, out: F64QrDecomposition, workspace: Workspace?): F64QrDecomposition {
        requireShape(out.m == a.rows && out.n == a.cols) {
            "qrInto: out is ${out.m}x${out.n}, expected ${a.rows}x${a.cols}"
        }
        val m = a.rows
        val n = a.cols
        a.data.copyInto(out.qr)
        if (m > 0 && n > 0) {
            val info = f.dgeqrf(COL_MAJOR, m, n, out.qr, m, out.tau)
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return out
    }

    override fun applyQInto(qr: F64QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
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

    /**
     * Ungated: `dgecon` scales its triangular solves against overflow where the portable estimator does not,
     * so it answers at every size a binding is present for. The estimate is what a caller consults before
     * trusting a solve, which is worth more than the call.
     */
    override fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        requireLuSquare(lu, "rcond")
        requireRcondAnorm(anorm)
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
        requireSquare(a, "cholesky")
        return choleskyInto(a, F64CholeskyDecomposition(F64DenseMatrix(a.rows, a.rows)), policy)
    }

    /** `dpotrf` works in place, so [out]'s buffer takes the lower triangle of [a] and is factored over.
     *  A non-positive pivot leaves it partly written before the portable path refactors into it. */
    override fun choleskyInto(
        a: F64DenseMatrix,
        out: F64CholeskyDecomposition,
        policy: CholeskyPolicy,
    ): F64CholeskyDecomposition {
        requireSquare(a, "cholesky")
        val n = a.rows
        requireShape(out.n == n) { "choleskyInto: out is ${out.n}x${out.n}, expected ${n}x$n" }
        if (n == 0) return out
        val ld = out.l.data
        // `dpotrf` may partly overwrite an aliased input before reporting the failure that sends it to the
        // portable policy path, so that exceptional aliasing case needs an original snapshot.
        val source = if (a.data === ld) a.data.copyOf() else a.data
        for (j in 0 until n) {
            // `dpotrf` never writes the strict upper triangle, which the factor promises is zero.
            for (i in 0 until j) ld[i + j * n] = 0.0
            source.copyInto(ld, j + j * n, j + j * n, (j + 1) * n)
        }
        val info = f.dpotrf(COL_MAJOR, LOWER_UPLO, n, ld, n)
        check(info >= 0) { "dpotrf: illegal argument ${-info}" }
        if (info > 0) {
            val original = if (source === a.data) a else F64DenseMatrix(n, n, source)
            return portable.choleskyInto(original, out, policy)
        }
        return out
    }

    /**
     * dpotri writes only the triangle it is given, so the result is mirrored, and it overwrites the factor,
     * so the factor is copied first.
     */
    override fun invert(chol: F64CholeskyDecomposition, workspace: Workspace?): F64DenseMatrix {
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
