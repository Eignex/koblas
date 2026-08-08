@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.MatrixLike
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.QrDecomposition
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.lapackFailedAt
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

// The CBLAS enums and LAPACKE layout macro, by their ABI integer values (the resolved function
// pointers declare the parameters as plain int; see OpenBlasBindings.kt).
private const val COL_MAJOR = 102
private const val NO_TRANS = 111
private const val TRANS = 112
private const val UPPER = 121
private const val LOWER = 122
private const val NON_UNIT = 131
private const val UNIT = 132
private const val LEFT = 141
private const val RIGHT = 142

/**
 * The host LAPACKE as koblas's [Lapack] half: the factorizations and the solves over their packed
 * formats.
 *
 * Needs [CblasFunctions] as well, because the blocked solves reach for `dtrsm` and `dtrsv` directly.
 * Only constructed when LAPACKE resolved; a host without it keeps the portable factorizations.
 */
internal class CblasLapack(private val f: LapackeFunctions, private val blas: CblasFunctions) : Lapack {
    override val name: String get() = "cblas"

    /** Above the reference (0). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv)
        val ipiv = IntArray(n)
        val info = lu.usePinned { lp ->
            ipiv.usePinned { pp -> f.dgetrf(COL_MAJOR, n, n, lp.addressOf(0), n, pp.addressOf(0)) }
        }
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

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        val n = lu.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
        if (n == 0) return out
        if (transpose) {
            // Aᵀ x = b, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ, back-solve unit Lᵀ, un-permute. The scatter
            // cannot run in place, so the solved vector is staged.
            val y = workspace?.take(n) ?: DoubleArray(n)
            b.copyInto(y)
            trsv(lu.lu, n, y, UPPER, TRANS, NON_UNIT)
            trsv(lu.lu, n, y, LOWER, TRANS, UNIT)
            for (i in 0 until n) out[lu.piv[i]] = y[i]
            workspace?.release(y)
        } else {
            // A x = b, with P A = L U: permute b, forward-solve unit L, back-solve U.
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
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (n == 0 || nrhs == 0) return out
        // Same permute + two block triangular solves as the vector path. dtrsm reads the buffers in their
        // native order, so nothing is transposed on the way in; the permutation is the strided step, one
        // element per column rather than a contiguous run per row.
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

    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
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

    /**
     * Delegates to the portable kernels, which are faster here at every size that fits in cache.
     *
     * Measured on this backend against the scalar kernels (us/op, linuxX64): n=64 5.3 versus 3.3, n=256
     * 107 versus 45, n=1024 1978 versus 2094. `dsytrs` only reaches parity at n=1024, and the routine is
     * `O(n^2)` work over `O(n^2)` data, so there is nothing to amortize the pivot-block bookkeeping
     * against. This is the one routine where the native library loses on this platform — every other
     * level-2 routine and the LU solves win by 2x to 15x, unlike on the JVM, whose SIMD kernels beat the
     * native library at level 2 outright. The blocked multi-RHS solve below stays native: there the work
     * grows with the right-hand-side count while the factor is read once.
     */
    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.solveInto(ldl, b, out)

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
        require(y.size == qr.m) { "applyQ: y length ${y.size} != ${qr.m}" }
        require(out.size == qr.m) { "applyQ: out length ${out.size} != ${qr.m}" }
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

    /** LAPACKE_dsytrs on [x] in place (lower, 1-based block ipiv); returns [x]. */
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
     * `dpotrf`, with two adjustments to match the koblas contract.
     *
     * LAPACK factorizes in place and leaves the strict upper triangle exactly as the input had it, while
     * koblas returns a factor whose upper triangle is zero, so it is cleared afterwards. And LAPACK has
     * no equivalent of [CholeskyPolicy.Regularize]: it reports the failing pivot instead of clamping it, so a
     * non-positive-definite input falls back to the portable path, which is what applies the clamp.
     */
    override fun cholesky(a: MatrixLike, policy: CholeskyPolicy): DenseMatrix {
        require(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (n == 0) return DenseMatrix(0, 0)
        val l = DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0..i) l[i, j] = a[i, j]
        val info = l.data.usePinned { lp -> f.dpotrf(COL_MAJOR, 'L'.code.toByte(), n, lp.addressOf(0), n) }
        check(info >= 0) { "dpotrf: illegal argument ${-info}" }
        // info > 0 marks the leading minor that is not positive definite: redo it portably, which either
        // clamps the pivot or throws, per the flag.
        if (info > 0) return super.cholesky(a, policy)
        for (i in 0 until n) for (j in i + 1 until n) l[i, j] = 0.0
        return l
    }

    /**
     * Delegates to the portable kernels, which win here by a wide and widening margin: measured on
     * linuxX64 (us/op, native against portable) 207 versus 66 at n=256, 6579 versus 740 at 1024, and
     * 64711 versus 5320 at 2048.
     *
     * The routine is `O(n^2)` work over `O(n^2)` data, so there is nothing to amortize a call against —
     * the same reason the LU and LDL vector solves delegate.
     *
     * Those margins were taken under row-major storage, where LAPACKE also transposed the matrix into a
     * column-major temporary on every call — 32 MB per solve at n=2048. koblas stores what LAPACK wants, so
     * that cost is not in the current numbers and these overstate the native side's disadvantage. Due a
     * re-measurement rather than settled; the JVM twin of this routine was re-measured and stayed portable.
     */
    override fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray = ReferenceLinearAlgebra.solveSpd(L, b)

    /**
     * `dpotri`, which writes only the triangle it is given; koblas returns the full symmetric inverse,
     * so the result is mirrored. The factor is copied first, since dpotri overwrites it with the inverse.
     */
    override fun invertSpd(L: DenseMatrix, workspace: Workspace?): DenseMatrix {
        val n = L.rows
        if (n == 0) return DenseMatrix(0, 0)
        val inv = DenseMatrix(n, n, L.data.copyOf())
        val info = inv.data.usePinned { ip -> f.dpotri(COL_MAJOR, 'L'.code.toByte(), n, ip.addressOf(0), n) }
        check(info >= 0) { "dpotri: illegal argument ${-info}" }
        check(info == 0) { "dpotri: zero diagonal at $info, factor is singular" }
        for (i in 0 until n) for (j in 0 until i) inv[j, i] = inv[i, j]
        return inv
    }

    /** cblas_dtrsv over the packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsv(a: DoubleArray, n: Int, x: DoubleArray, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            x.usePinned { xp ->
                blas.dtrsv(COL_MAJOR, uplo, trans, diag, n, ap.addressOf(0), n, xp.addressOf(0), 1)
            }
        }
    }

    /** Left-side cblas_dtrsm over the packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsmLeft(a: DoubleArray, n: Int, b: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            b.usePinned { bp ->
                blas.dtrsm(COL_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0, ap.addressOf(0), n, bp.addressOf(0), n)
            }
        }
    }
}
