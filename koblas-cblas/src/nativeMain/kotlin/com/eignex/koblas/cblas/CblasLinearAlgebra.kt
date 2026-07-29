@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LdlDecomposition
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.QrDecomposition
import com.eignex.koblas.Uplo
import com.eignex.koblas.Workspace
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

// The CBLAS enums and LAPACKE layout macro, by their ABI integer values (the resolved function
// pointers declare the parameters as plain int; see OpenBlasBindings.kt).
private const val ROW_MAJOR = 101
private const val NO_TRANS = 111
private const val TRANS = 112
private const val UPPER = 121
private const val LOWER = 122
private const val NON_UNIT = 131
private const val UNIT = 132
private const val LEFT = 141
private const val RIGHT = 142

/**
 * [LinearAlgebra] backed by the host's OpenBLAS through its C interfaces (CBLAS and LAPACKE), for
 * the Linux and macOS native targets. Nothing is linked: the libraries are resolved with `dlopen`
 * at program start, so the dependency is optional at runtime — `libopenblas` plus `liblapacke` on
 * Debian/Ubuntu, `brew install openblas` on macOS. When they are present, depending on the
 * koblas-cblas artifact installs this backend eagerly before `main`; when they are missing the
 * program still runs on [com.eignex.koblas.ReferenceLinearAlgebra], and constructing this class
 * throws. [isAvailable] reports which case the host is.
 *
 * Koblas storage is row-major, which CBLAS and LAPACKE support directly, so buffers cross the FFI
 * boundary without repacking. Semantics match [com.eignex.koblas.ReferenceLinearAlgebra] exactly as
 * specified by the [LinearAlgebra] contract: `beta == 0` overwrites without reading, `alpha == 0`
 * reduces to the `beta` scale, [syrk] produces the full, exactly symmetric result by default, and
 * the factorizations use the shared packed formats so they interchange between backends.
 *
 * OpenBLAS runs single-threaded by default here, which is the faster configuration at koblas
 * workload sizes; set the `OPENBLAS_NUM_THREADS` environment variable to opt into its threading.
 */
class CblasLinearAlgebra : LinearAlgebra {
    /** Host-availability check for the backend. */
    companion object {
        /** Whether the host provides the OpenBLAS (and LAPACKE) symbols this backend needs. */
        fun isAvailable(): Boolean = OpenBlasLoader.functions != null
    }

    private val f = requireNotNull(OpenBlasLoader.functions) {
        "OpenBLAS is not available on this host; koblas falls back to the reference backend"
    }

    override val name: String get() = "cblas"

    /** Above the reference (0), below koblas-openblas's bundled natives (100). */
    override val priority: Int get() = 90

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        require(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        require(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        // Degenerate cases short-circuit in Kotlin: the BLAS quick-return paths do not honor the
        // beta == 0 overwrite when a dimension is zero.
        if (alpha == 0.0 || xLen == 0) {
            scaleInPlace(y, beta)
            return
        }
        if (yLen == 0) return
        val trans = if (transpose) TRANS else NO_TRANS
        a.data.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    f.dgemv(
                        ROW_MAJOR, trans, a.rows, a.cols, alpha,
                        ap.addressOf(0), a.cols, xp.addressOf(0), 1, beta, yp.addressOf(0), 1,
                    )
                }
            }
        }
    }

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
        a.data.usePinned { ap ->
            b.data.usePinned { bp ->
                c.data.usePinned { cp ->
                    f.dgemm(
                        ROW_MAJOR, transA, transB, m, n, k, alpha,
                        ap.addressOf(0), a.cols, bp.addressOf(0), b.cols, beta, cp.addressOf(0), c.cols,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList", "ReturnCount") // the BLAS dsyrk signature; guard-clause style
    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, uplo: Uplo) {
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
            a.data.usePinned { ap ->
                c.data.usePinned { cp ->
                    f.dsyrk(ROW_MAJOR, u, trans, n, k, alpha, ap.addressOf(0), a.cols, beta, cp.addressOf(0), n)
                }
            }
            return
        }
        // dsyrk touches one triangle only, while the FULL contract promises the full, exactly
        // symmetric alpha term on top of a beta scale of all of C. Compute the alpha term into a
        // scratch lower triangle, mirror it, then combine.
        val w = DoubleArray(n * n)
        a.data.usePinned { ap ->
            w.usePinned { wp ->
                f.dsyrk(ROW_MAJOR, LOWER, trans, n, k, alpha, ap.addressOf(0), a.cols, 0.0, wp.addressOf(0), n)
            }
        }
        for (i in 0 until n) {
            for (j in 0 until i) w[j * n + i] = w[i * n + j]
        }
        val cd = c.data
        when (beta) {
            0.0 -> w.copyInto(cd)

            else -> {
                if (beta != 1.0) cd.usePinned { cp -> f.dscal(cd.size, beta, cp.addressOf(0), 1) }
                w.usePinned { wp ->
                    cd.usePinned { cp -> f.daxpy(cd.size, 1.0, wp.addressOf(0), 1, cp.addressOf(0), 1) }
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        require(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        require(x.size == n) { "symv: x length ${x.size} != $n" }
        require(y.size == n) { "symv: y length ${y.size} != $n" }
        if (alpha == 0.0) {
            scaleInPlace(y, beta)
            return
        }
        if (n == 0) return
        val uplo = if (lower) LOWER else UPPER
        a.data.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    f.dsymv(
                        ROW_MAJOR, uplo, n, alpha,
                        ap.addressOf(0), n, xp.addressOf(0), 1, beta, yp.addressOf(0), 1,
                    )
                }
            }
        }
    }

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
        a.data.usePinned { ap ->
            b.data.usePinned { bp ->
                c.data.usePinned { cp ->
                    f.dsymm(
                        ROW_MAJOR, side, uplo, c.rows, c.cols, alpha,
                        ap.addressOf(0), m, bp.addressOf(0), c.cols, beta, cp.addressOf(0), c.cols,
                    )
                }
            }
        }
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv, singular = false)
        val ipiv = IntArray(n)
        val info = lu.usePinned { lp ->
            ipiv.usePinned { pp -> f.dgetrf(ROW_MAJOR, n, n, lp.addressOf(0), n, pp.addressOf(0)) }
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
        return LuDecomposition(n, lu, piv, singular = info > 0)
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
            val y = workspace?.backendVector(n) ?: DoubleArray(n)
            b.copyInto(y)
            trsv(lu.lu, n, y, UPPER, TRANS, NON_UNIT)
            trsv(lu.lu, n, y, LOWER, TRANS, UNIT)
            for (i in 0 until n) out[lu.piv[i]] = y[i]
        } else {
            // A x = b, with P A = L U: permute b, forward-solve unit L, back-solve U.
            if (out === b) {
                val staged = workspace?.backendVector(n) ?: DoubleArray(n)
                for (i in 0 until n) staged[i] = b[lu.piv[i]]
                staged.copyInto(out)
            } else {
                for (i in 0 until n) out[i] = b[lu.piv[i]]
            }
            trsv(lu.lu, n, out, LOWER, NO_TRANS, UNIT)
            trsv(lu.lu, n, out, UPPER, NO_TRANS, NON_UNIT)
        }
        return out
    }

    override fun solve(lu: LuDecomposition, b: DenseMatrix, transpose: Boolean): DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        if (n == 0 || nrhs == 0) return DenseMatrix.wrap(n, nrhs, DoubleArray(n * nrhs))
        // Same permute + two block triangular solves as the vector path; dtrsm is row-major native,
        // so no LAPACKE transposition tax scales with nrhs.
        return if (transpose) {
            val y = b.data.copyOf()
            trsmLeft(lu.lu, n, y, nrhs, UPPER, TRANS, NON_UNIT)
            trsmLeft(lu.lu, n, y, nrhs, LOWER, TRANS, UNIT)
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) y.copyInto(x, lu.piv[i] * nrhs, i * nrhs, (i + 1) * nrhs)
            DenseMatrix.wrap(n, nrhs, x)
        } else {
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) b.data.copyInto(x, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
            trsmLeft(lu.lu, n, x, nrhs, LOWER, NO_TRANS, UNIT)
            trsmLeft(lu.lu, n, x, nrhs, UPPER, NO_TRANS, NON_UNIT)
            DenseMatrix.wrap(n, nrhs, x)
        }
    }

    override fun ldl(a: DenseMatrix): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv, singular = false)
        val info = buf.usePinned { bp ->
            ipiv.usePinned { pp ->
                f.dsytrf(ROW_MAJOR, 'L'.code.toByte(), n, bp.addressOf(0), n, pp.addressOf(0))
            }
        }
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, singular = info > 0)
    }

    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
        if (n == 0) return out
        if (out !== b) b.copyInto(out)
        return solveSytrs(ldl, out, 1)
    }

    override fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        val x = b.data.copyOf()
        if (n == 0 || nrhs == 0) return DenseMatrix.wrap(n, nrhs, x)
        return DenseMatrix.wrap(n, nrhs, solveSytrs(ldl, x, nrhs))
    }

    override fun qr(a: DenseMatrix): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = buf.usePinned { bp ->
                tau.usePinned { tp -> f.dgeqrf(ROW_MAJOR, m, n, bp.addressOf(0), n, tp.addressOf(0)) }
            }
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
        val info = qr.qr.usePinned { qp ->
            qr.tau.usePinned { tp ->
                c.usePinned { cp ->
                    f.dormqr(
                        ROW_MAJOR, side, trans, qr.m, 1, qr.tau.size,
                        qp.addressOf(0), qr.n, tp.addressOf(0), cp.addressOf(0), 1,
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
                f.dgecon(ROW_MAJOR, '1'.code.toByte(), n, lp.addressOf(0), n, anorm, op.addressOf(0))
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
                        ROW_MAJOR, 'L'.code.toByte(), n, nrhs,
                        lp.addressOf(0), n, pp.addressOf(0), xp.addressOf(0), nrhs,
                    )
                }
            }
        }
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return x
    }

    /** cblas_dtrsv over the packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsv(a: DoubleArray, n: Int, x: DoubleArray, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            x.usePinned { xp ->
                f.dtrsv(ROW_MAJOR, uplo, trans, diag, n, ap.addressOf(0), n, xp.addressOf(0), 1)
            }
        }
    }

    /** Left-side cblas_dtrsm over the packed factor buffer. */
    @Suppress("LongParameterList")
    private fun trsmLeft(a: DoubleArray, n: Int, b: DoubleArray, nrhs: Int, uplo: Int, trans: Int, diag: Int) {
        a.usePinned { ap ->
            b.usePinned { bp ->
                f.dtrsm(ROW_MAJOR, LEFT, uplo, trans, diag, n, nrhs, 1.0, ap.addressOf(0), n, bp.addressOf(0), nrhs)
            }
        }
    }

    /** `v = beta * v` honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)
            beta != 1.0 && v.isNotEmpty() -> v.usePinned { vp -> f.dscal(v.size, beta, vp.addressOf(0), 1) }
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
