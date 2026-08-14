@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.CholeskyDecomposition
import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.QrDecomposition
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.lapackFailedAt
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

// The CBLAS enums and the LAPACKE layout macro by their ABI integer values, declared as plain int.
private const val COL_MAJOR = 102
private const val NO_TRANS = 111
private const val TRANS = 112
private const val UPPER = 121
private const val LOWER = 122
private const val NON_UNIT = 131
private const val UNIT = 132
private const val LEFT = 141
private const val RIGHT = 142

/** Constructed only when LAPACKE resolved, so a host without it keeps the portable factorizations. */
internal class CblasLapack(private val f: LapackeFunctions, private val blas: CblasFunctions) : Lapack {
    override val name: String get() = "cblas"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override fun factor(a: DenseMatrix): LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return factorInto(a, LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    /** `dgetrf` works in place, so [out]'s buffers take the copy of [a] and the factorization overwrites it. */
    override fun factorInto(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        val n = out.n
        a.data.copyInto(out.lu)
        val piv = out.piv
        for (i in 0 until n) piv[i] = i
        out.failedAt = NOT_SINGULAR
        if (n == 0) return out
        val ipiv = IntArray(n)
        val info = out.lu.usePinned { lp ->
            ipiv.usePinned { pp -> f.dgetrf(COL_MAJOR, n, n, lp.addressOf(0), n, pp.addressOf(0)) }
        }
        check(info >= 0) { "dgetrf: illegal argument ${-info}" }
        // dgetrf reports successive 1-based row swaps; replaying them gives the piv form LuDecomposition uses.
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

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
        requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
        if (n == 0) return out
        if (transpose) {
            // Aᵀ = Uᵀ Lᵀ P, so forward-solve Uᵀ, back-solve unit Lᵀ, un-permute.
            val y = workspace?.take(n) ?: DoubleArray(n)
            b.copyInto(y)
            trsv(lu.lu, n, y, UPPER, TRANS, NON_UNIT)
            trsv(lu.lu, n, y, LOWER, TRANS, UNIT)
            for (i in 0 until n) out[lu.piv[i]] = y[i]
            workspace?.release(y)
        } else {
            // P A = L U, so permute b, forward-solve unit L, back-solve U.
            if (out === b) {
                val staged = workspace?.take(n) ?: DoubleArray(n)
                for (i in 0 until n) staged[i] = b[lu.piv[i]]
                staged.copyInto(out)
                workspace?.release(staged)
            } else {
                for (i in 0 until n) out[i] = b[lu.piv[i]]
            }
            trsv(lu.lu, n, out, LOWER, NO_TRANS, UNIT)
            trsv(lu.lu, n, out, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

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
        if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmLeft(lu.lu, n, y, nrhs, UPPER, TRANS, NON_UNIT)
            trsmLeft(lu.lu, n, y, nrhs, LOWER, TRANS, UNIT)
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
            trsmLeft(lu.lu, n, out.data, nrhs, LOWER, NO_TRANS, UNIT)
            trsmLeft(lu.lu, n, out.data, nrhs, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

    /** Writes dst(i, c) = src(piv(i), c) when [gather], its inverse otherwise, over an n by nrhs pair. */
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

    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        requireShape(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv)
        val info = buf.usePinned { bp ->
            ipiv.usePinned { pp ->
                f.dsytrf(COL_MAJOR, 'L'.code.toByte(), n, bp.addressOf(0), n, pp.addressOf(0))
            }
        }
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, lapackFailedAt(info))
    }

    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.solveInto(ldl, b, out)

    override fun solveInto(
        ldl: LdlDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        workspace: Workspace?,
    ): DenseMatrix {
        requireFactored(ldl.failedAt, "solve")
        val n = ldl.n
        val nrhs = b.cols
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        requireShape(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (out !== b) b.data.copyInto(out.data)
        if (n == 0 || nrhs == 0) return out
        solveSytrs(ldl, out.data, nrhs)
        return out
    }

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = buf.usePinned { bp ->
                tau.usePinned { tp -> f.dgeqrf(COL_MAJOR, m, n, bp.addressOf(0), m, tp.addressOf(0)) }
            }
            check(info == 0) { "dgeqrf: illegal argument ${-info}" }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    override fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        requireShape(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        requireShape(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
        val c = out
        if (out !== y) y.copyInto(out)
        if (qr.tau.isEmpty()) return c
        val side = 'L'.code.toByte()
        val trans = (if (transpose) 'T' else 'N').code.toByte()
        val info = qr.qr.usePinned { qp ->
            qr.tau.usePinned { tp ->
                c.usePinned { cp ->
                    f.dormqr(
                        COL_MAJOR, side, trans, qr.m, 1, qr.tau.size,
                        qp.addressOf(0), qr.m, tp.addressOf(0), cp.addressOf(0), qr.m,
                    )
                }
            }
        }
        check(info == 0) { "dormqr: illegal argument ${-info}" }
        return c
    }

    override fun rcond(lu: LuDecomposition, anorm: Double, workspace: Workspace?): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val out = DoubleArray(1)
        val info = lu.lu.usePinned { lp ->
            out.usePinned { op ->
                f.dgecon(COL_MAJOR, '1'.code.toByte(), n, lp.addressOf(0), n, anorm, op.addressOf(0))
            }
        }
        check(info == 0) { "dgecon: illegal argument ${-info}" }
        return out[0]
    }

    /** Runs LAPACKE_dsytrs on [x] in place, lower triangle with 1-based block ipiv. */
    private fun solveSytrs(ldl: LdlDecomposition, x: DoubleArray, nrhs: Int): DoubleArray {
        val n = ldl.n
        val info = ldl.ldl.usePinned { lp ->
            ldl.ipiv.usePinned { pp ->
                x.usePinned { xp ->
                    f.dsytrs(
                        COL_MAJOR, 'L'.code.toByte(), n, nrhs,
                        lp.addressOf(0), n, pp.addressOf(0), xp.addressOf(0), n,
                    )
                }
            }
        }
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return x
    }

    /**
     * dpotrf leaves the strict upper triangle as the input had it, so it is cleared to match the koblas
     * factor, and it has no [CholeskyPolicy.Regularize] equivalent, so that case falls back to portable.
     */
    override fun cholesky(a: DenseMatrix, policy: CholeskyPolicy): CholeskyDecomposition {
        requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (n == 0) return CholeskyDecomposition(DenseMatrix(0, 0))
        val l = DenseMatrix(n, n)
        // One bulk copy per column of the lower triangle, which is as far as dpotrf reads.
        for (j in 0 until n) a.data.copyInto(l.data, j + j * n, j + j * n, (j + 1) * n)
        val info = l.data.usePinned { lp -> f.dpotrf(COL_MAJOR, 'L'.code.toByte(), n, lp.addressOf(0), n) }
        check(info >= 0) { "dpotrf: illegal argument ${-info}" }
        // info > 0 marks the leading minor that is not positive definite.
        if (info > 0) return super.cholesky(a, policy)
        for (i in 0 until n) for (j in i + 1 until n) l[i, j] = 0.0
        return CholeskyDecomposition(l)
    }

    override fun solve(chol: CholeskyDecomposition, b: DoubleArray): DoubleArray = ReferenceLinearAlgebra.solve(chol, b)

    /** dpotri writes one triangle and overwrites its input, so the factor is copied and then mirrored. */
    override fun invert(chol: CholeskyDecomposition, workspace: Workspace?): DenseMatrix {
        val n = chol.n
        if (n == 0) return DenseMatrix(0, 0)
        val inv = DenseMatrix(n, n, chol.l.data.copyOf())
        val info = inv.data.usePinned { ip -> f.dpotri(COL_MAJOR, 'L'.code.toByte(), n, ip.addressOf(0), n) }
        check(info >= 0) { "dpotri: illegal argument ${-info}" }
        check(info == 0) { "dpotri: zero diagonal at $info, factor is singular" }
        for (i in 0 until n) for (j in 0 until i) inv[j, i] = inv[i, j]
        return inv
    }

    @Suppress("LongParameterList")
    private fun trsv(a: DoubleArray, n: Int, x: DoubleArray, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            x.usePinned { xp ->
                blas.dtrsv(COL_MAJOR, uplo, trans, diag, n, ap.addressOf(0), n, xp.addressOf(0), 1)
            }
        }
    }

    @Suppress("LongParameterList")
    private fun trsmLeft(a: DoubleArray, n: Int, b: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            b.usePinned { bp ->
                blas.dtrsm(COL_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0, ap.addressOf(0), n, bp.addressOf(0), n)
            }
        }
    }
}
