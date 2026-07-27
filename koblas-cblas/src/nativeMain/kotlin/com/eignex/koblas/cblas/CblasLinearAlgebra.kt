package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LdlDecomposition
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.QrDecomposition
import com.eignex.koblas.cblas.capi.LAPACKE_dgecon
import com.eignex.koblas.cblas.capi.LAPACKE_dgeqrf
import com.eignex.koblas.cblas.capi.LAPACKE_dgetrf
import com.eignex.koblas.cblas.capi.LAPACKE_dormqr
import com.eignex.koblas.cblas.capi.LAPACKE_dsytrf
import com.eignex.koblas.cblas.capi.LAPACKE_dsytrs
import com.eignex.koblas.cblas.capi.cblas_daxpy
import com.eignex.koblas.cblas.capi.cblas_dgemm
import com.eignex.koblas.cblas.capi.cblas_dgemv
import com.eignex.koblas.cblas.capi.cblas_dscal
import com.eignex.koblas.cblas.capi.cblas_dsymm
import com.eignex.koblas.cblas.capi.cblas_dsymv
import com.eignex.koblas.cblas.capi.cblas_dsyrk
import com.eignex.koblas.cblas.capi.cblas_dtrsv
import com.eignex.koblas.cblas.capi.openblas_set_num_threads
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.refTo
import platform.posix.getenv

// The CBLAS enums and LAPACKE layout macro, by their ABI integer values (the cinterop declares the
// parameters as plain int; see cblas.def).
private const val ROW_MAJOR = 101
private const val NO_TRANS = 111
private const val TRANS = 112
private const val UPPER = 121
private const val LOWER = 122
private const val NON_UNIT = 131
private const val UNIT = 132
private const val LEFT = 141

/**
 * [LinearAlgebra] backed by the system-installed OpenBLAS through its C interfaces (CBLAS and
 * LAPACKE), for the Linux and macOS native targets. Unlike koblas-openblas on the JVM, nothing is
 * bundled: the library links `libopenblas` (and, on Linux, `liblapacke`) from the host —
 * `libopenblas-dev` and `liblapacke-dev` on Debian/Ubuntu, `brew install openblas` on macOS.
 *
 * Koblas storage is row-major, which CBLAS and LAPACKE support directly, so buffers cross the FFI
 * boundary without repacking. Semantics match [com.eignex.koblas.ReferenceLinearAlgebra] exactly as
 * specified by the [LinearAlgebra] contract: `beta == 0` overwrites without reading, `alpha == 0`
 * reduces to the `beta` scale, [syrk] produces the full, exactly symmetric result, and the
 * factorizations use the shared packed formats so they interchange between backends.
 *
 * OpenBLAS runs single-threaded by default here, which is the faster configuration at koblas
 * workload sizes; set the `OPENBLAS_NUM_THREADS` environment variable to opt into its threading.
 *
 * Native targets have no runtime discovery, so activate the backend explicitly once at startup:
 * `installLinearAlgebra(CblasLinearAlgebra())`.
 */
@OptIn(ExperimentalForeignApi::class)
class CblasLinearAlgebra : LinearAlgebra {
    /** Applies the threading default once, when the first instance loads the class. */
    companion object {
        init {
            // An explicit environment override is honored by OpenBLAS itself.
            if (getenv("OPENBLAS_NUM_THREADS") == null) openblas_set_num_threads(1)
        }
    }

    override val name: String get() = "cblas"

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
        cblas_dgemv(
            ROW_MAJOR, trans, a.rows, a.cols, alpha,
            a.data.refTo(0), a.cols, x.refTo(0), 1, beta, y.refTo(0), 1,
        )
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
        cblas_dgemm(
            ROW_MAJOR, transA, transB, m, n, k,
            alpha, a.data.refTo(0), a.cols, b.data.refTo(0), b.cols, beta, c.data.refTo(0), c.cols,
        )
    }

    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        require(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (alpha == 0.0 || k == 0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (n == 0) return
        // dsyrk touches one triangle only, while the contract promises the full, exactly symmetric
        // alpha term on top of a beta scale of all of C. Compute the alpha term into a scratch lower
        // triangle, mirror it, then combine.
        val w = DoubleArray(n * n)
        val trans = if (transpose) TRANS else NO_TRANS
        cblas_dsyrk(ROW_MAJOR, LOWER, trans, n, k, alpha, a.data.refTo(0), a.cols, 0.0, w.refTo(0), n)
        for (i in 0 until n) {
            for (j in 0 until i) w[j * n + i] = w[i * n + j]
        }
        val cd = c.data
        when (beta) {
            0.0 -> w.copyInto(cd)

            else -> {
                if (beta != 1.0) cblas_dscal(cd.size, beta, cd.refTo(0), 1)
                cblas_daxpy(cd.size, 1.0, w.refTo(0), 1, cd.refTo(0), 1)
            }
        }
    }

    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray) {
        require(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        require(x.size == n) { "symv: x length ${x.size} != $n" }
        require(y.size == n) { "symv: y length ${y.size} != $n" }
        if (alpha == 0.0) {
            scaleInPlace(y, beta)
            return
        }
        if (n == 0) return
        cblas_dsymv(ROW_MAJOR, LOWER, n, alpha, a.data.refTo(0), n, x.refTo(0), 1, beta, y.refTo(0), 1)
    }

    override fun symm(alpha: Double, a: DenseMatrix, b: DenseMatrix, beta: Double, c: DenseMatrix) {
        require(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        val m = a.rows
        val p = b.cols
        require(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        require(c.rows == m && c.cols == p) { "symm: C is ${c.rows}x${c.cols}, expected ${m}x$p" }
        if (alpha == 0.0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (m == 0 || p == 0) return
        cblas_dsymm(
            ROW_MAJOR, LEFT, LOWER, m, p, alpha,
            a.data.refTo(0), m, b.data.refTo(0), p, beta, c.data.refTo(0), p,
        )
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        if (n == 0) return LuDecomposition(0, lu, piv, singular = false)
        val ipiv = IntArray(n)
        val info = LAPACKE_dgetrf(ROW_MAJOR, n, n, lu.refTo(0), n, ipiv.refTo(0))
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

    override fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean): DoubleArray {
        val n = lu.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        if (n == 0) return DoubleArray(0)
        val f = lu.lu
        return if (transpose) {
            // Aᵀ x = b, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ, back-solve unit Lᵀ, un-permute.
            val y = b.copyOf()
            cblas_dtrsv(ROW_MAJOR, UPPER, TRANS, NON_UNIT, n, f.refTo(0), n, y.refTo(0), 1)
            cblas_dtrsv(ROW_MAJOR, LOWER, TRANS, UNIT, n, f.refTo(0), n, y.refTo(0), 1)
            val x = DoubleArray(n)
            for (i in 0 until n) x[lu.piv[i]] = y[i]
            x
        } else {
            // A x = b, with P A = L U: permute b, forward-solve unit L, back-solve U.
            val x = DoubleArray(n) { b[lu.piv[it]] }
            cblas_dtrsv(ROW_MAJOR, LOWER, NO_TRANS, UNIT, n, f.refTo(0), n, x.refTo(0), 1)
            cblas_dtrsv(ROW_MAJOR, UPPER, NO_TRANS, NON_UNIT, n, f.refTo(0), n, x.refTo(0), 1)
            x
        }
    }

    override fun ldl(a: DenseMatrix): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val buf = a.data.copyOf()
        val ipiv = IntArray(n)
        if (n == 0) return LdlDecomposition(0, buf, ipiv, singular = false)
        val info = LAPACKE_dsytrf(ROW_MAJOR, 'L'.code.toByte(), n, buf.refTo(0), n, ipiv.refTo(0))
        check(info >= 0) { "dsytrf: illegal argument ${-info}" }
        return LdlDecomposition(n, buf, ipiv, singular = info > 0)
    }

    override fun solve(ldl: LdlDecomposition, b: DoubleArray): DoubleArray {
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        if (n == 0) return DoubleArray(0)
        val x = b.copyOf()
        val info = LAPACKE_dsytrs(
            ROW_MAJOR, 'L'.code.toByte(), n, 1,
            ldl.ldl.refTo(0), n, ldl.ipiv.refTo(0), x.refTo(0), 1,
        )
        check(info == 0) { "dsytrs: illegal argument ${-info}" }
        return x
    }

    override fun qr(a: DenseMatrix): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        if (m > 0 && n > 0) {
            val info = LAPACKE_dgeqrf(ROW_MAJOR, m, n, buf.refTo(0), n, tau.refTo(0))
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
        val info = LAPACKE_dormqr(
            ROW_MAJOR, side, trans, qr.m, 1, qr.tau.size,
            qr.qr.refTo(0), qr.n, qr.tau.refTo(0), c.refTo(0), 1,
        )
        check(info == 0) { "dormqr: illegal argument ${-info}" }
        return c
    }

    override fun rcond(lu: LuDecomposition, anorm: Double): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val out = DoubleArray(1)
        val info = LAPACKE_dgecon(ROW_MAJOR, '1'.code.toByte(), n, lu.lu.refTo(0), n, anorm, out.refTo(0))
        check(info == 0) { "dgecon: illegal argument ${-info}" }
        return out[0]
    }

    /** `v = beta * v` honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)
            beta != 1.0 && v.isNotEmpty() -> cblas_dscal(v.size, beta, v.refTo(0), 1)
        }
    }
}
