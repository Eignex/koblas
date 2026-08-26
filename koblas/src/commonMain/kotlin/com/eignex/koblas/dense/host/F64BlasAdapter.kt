@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.cblas.Cblas.LOWER
import com.eignex.koblas.dense.host.cblas.Cblas.UPPER
import com.eignex.koblas.dense.host.cblas.Cblas.diagOf
import com.eignex.koblas.dense.host.cblas.Cblas.sideOf
import com.eignex.koblas.dense.host.cblas.Cblas.transOf
import com.eignex.koblas.dense.host.cblas.Cblas.uploOf
import com.eignex.koblas.internal.backend.DispatchThresholds
import com.eignex.koblas.internal.backend.dispatchesLevel3

/**
 * The dense matrix routines a host CBLAS provides, over whichever [CblasCalls] the platform supplies. Both
 * host bindings are this class plus their own FFI mechanism.
 *
 * Below the [DispatchThresholds] gate for its level, each routine hands back to the portable
 * implementation: crossing into a native library costs more than the work saves on a small problem.
 */
@Suppress("TooManyFunctions") // the BLAS surface a host library covers
public abstract class F64BlasAdapter internal constructor(
    private val f: CblasCalls,
    private val portable: F64ReferenceBlas = F64ReferenceBlas(),
    private val dispatch: DispatchThresholds = DispatchThresholds(0, 0, 0, 0),
) : F64Blas {

    /** A binding that calls out, whatever the portable instance it falls back to reports. */
    override val isPortable: Boolean get() = false

    // The routines below have no host binding, so they run the portable versions. Forwarded explicitly
    // rather than by class delegation, which would route a caller's convenience overloads to the portable
    // routine instead of the accelerated one, since a delegated member calls back into the delegate.
    // A transpose is `omatcopy`, a BLAS-like extension rather than part of CBLAS. OpenBLAS does export
    // cblas_domatcopy, and binding it was measured against the portable loop on the JVM: it takes about a
    // third off a 16 or a 64, which is a matter of microseconds, and from 256 up the two sit inside the
    // run-to-run spread whichever way the shape runs, since a transpose that size is bound by memory rather
    // than by the loop. The advantage therefore shrinks with size instead of growing, leaving no dimension
    // from which the call starts paying, which is the only shape a dispatch gate can express.
    override fun transpose(a: F64DenseMatrix): F64DenseMatrix = portable.transpose(a)

    override fun syr(alpha: Double, x: F64VectorLike, a: F64DenseMatrix, uplo: Uplo): Unit = portable.syr(
        alpha,
        x,
        a,
        uplo,
    )

    override fun syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, a: F64DenseMatrix, uplo: Uplo): Unit =
        portable.syr2(alpha, x, y, a, uplo)

    @Suppress("LongParameterList") // the BLAS dsyr2k signature plus optional scratch
    override fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ): Unit = portable.syr2k(alpha, a, b, transpose, beta, c, uplo, workspace)

    /** `v = beta * v`, honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)
            beta != 1.0 && v.isNotEmpty() -> f.dscal(v.size, beta, v, 1)
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix) {
        if (minOf(a.rows, a.cols) < dispatch.level2) return portable.ger(alpha, x, y, a)
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        f.dger(COL_MAJOR, a.rows, a.cols, alpha, x, 1, y, 1, a.data, a.rows)
    }

    override fun trsv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows < dispatch.level2) return portable.trsv(a, x, lower, transpose, unitDiag)
        nativeTriangularVector(a, x, lower, transpose, unitDiag, solve = true)
    }

    override fun trmv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows < dispatch.level2) return portable.trmv(a, x, lower, transpose, unitDiag)
        nativeTriangularVector(a, x, lower, transpose, unitDiag, solve = false)
    }

    /** dtrsv and dtrmv take the same arguments and differ only in the entry point, as dtrsm and dtrmm do. */
    @Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
    private fun nativeTriangularVector(
        a: F64DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        solve: Boolean,
    ) {
        val what = if (solve) "trsv" else "trmv"
        requireSquare(a, what)
        requireShape(x.size == a.rows) { "$what: x length ${x.size} != ${a.rows}" }
        if (a.rows == 0) return
        val call = if (solve) f::dtrsv else f::dtrmv
        call(COL_MAJOR, uploOf(lower), transOf(transpose), diagOf(unitDiag), a.rows, a.data, a.rows, x, 1)
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        if (minOf(a.rows, b.rows, b.cols) < dispatch.level3) {
            return portable.trsm(a, b, lower, transpose, unitDiag, right, alpha)
        }
        nativeTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true)
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        if (minOf(a.rows, b.rows, b.cols) < dispatch.level3) {
            return portable.trmm(a, b, lower, transpose, unitDiag, right, alpha)
        }
        nativeTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false)
    }

    /** dtrsm and dtrmm take the same arguments and differ only in the entry point. */
    @Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
    private fun nativeTriangularMatrix(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireSquare(a, what)
        if (right) {
            requireShape(b.cols == a.rows) { "$what right: B has ${b.cols} cols, expected ${a.rows}" }
        } else {
            requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
        }
        if (a.rows == 0 || b.rows == 0 || b.cols == 0) return
        val call = if (solve) f::dtrsm else f::dtrmm
        call(
            COL_MAJOR, sideOf(right), uploOf(lower), transOf(transpose), diagOf(unitDiag),
            b.rows, b.cols, alpha, a.data, a.rows, b.data, b.rows,
        )
    }

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        if (minOf(a.rows, a.cols) < dispatch.level2) {
            return portable.gemv(alpha, a, x, beta, y, transpose)
        }
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        // The BLAS quick-return paths do not honor the beta == 0 overwrite when a dimension is zero.
        if (alpha == 0.0 || xLen == 0) {
            scaleInPlace(y, beta)
            return
        }
        if (yLen == 0) return
        f.dgemv(COL_MAJOR, transOf(transpose), a.rows, a.cols, alpha, a.data, a.rows, x, 1, beta, y, 1)
    }

    override fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        if (alpha == 0.0 || k == 0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (m == 0 || n == 0) return
        if (!dispatchesLevel3(dispatch.level3, m, n, k)) {
            return portable.gemm(alpha, a, transposeA, b, transposeB, beta, c)
        }
        f.dgemm(
            COL_MAJOR, transOf(transposeA), transOf(transposeB), m, n, k, alpha,
            a.data, a.rows, b.data, b.rows, beta, c.data, c.rows,
        )
    }

    @Suppress("LongParameterList", "ReturnCount") // dsyrk's arguments plus scratch; guard-clause style
    override fun syrk(
        alpha: Double,
        a: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (minOf(n, k) < dispatch.level3) {
            return portable.syrk(alpha, a, transpose, beta, c, uplo, workspace)
        }
        if (alpha == 0.0 || k == 0) {
            scaleUplo(kernels, c.data, n, beta, uplo)
            return
        }
        if (n == 0) return
        val trans = transOf(transpose)
        if (uplo != Uplo.FULL) {
            // Strict dsyrk semantics: one triangle written and beta-scaled, the other untouched.
            f.dsyrk(
                COL_MAJOR,
                if (uplo ==
                    Uplo.LOWER
                ) {
                    LOWER
                } else {
                    UPPER
                },
                trans, n, k, alpha, a.data, a.rows, beta, c.data, n,
            )
            return
        }
        // dsyrk touches one triangle only, so the FULL contract needs the alpha term computed into a
        // scratch lower triangle, mirrored, then combined with a beta scale of all of C.
        val w = workspace?.take(n * n) ?: DoubleArray(n * n)
        // Handed back even when the library raises: a buffer never returned is one its pool can neither lend
        // again nor reclaim, and every call through here is a call that can fail.
        try {
            f.dsyrk(COL_MAJOR, LOWER, trans, n, k, alpha, a.data, a.rows, 0.0, w, n)
            for (j in 0 until n) {
                for (i in j + 1 until n) w[j + i * n] = w[i + j * n]
            }
            val cd = c.data
            if (beta == 0.0) {
                w.copyInto(cd)
            } else {
                if (beta != 1.0) f.dscal(cd.size, beta, cd, 1)
                f.daxpy(cd.size, 1.0, w, 1, cd, 1)
            }
        } finally {
            workspace?.release(w)
        }
    }

    /**
     * The shape is checked ahead of the gate: `dsymv` takes one dimension and a leading dimension, so a
     * non-square matrix would have it read `n²` entries from a shorter array, past the end of the buffer.
     */
    override fun symv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        if (a.rows < dispatch.level2) return portable.symv(alpha, a, x, beta, y, lower)
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
        if (alpha == 0.0 || n == 0) {
            scaleInPlace(y, beta)
            return
        }
        f.dsymv(COL_MAJOR, uploOf(lower), n, alpha, a.data, n, x, 1, beta, y, 1)
    }

    override fun symm(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        if (minOf(a.rows, b.rows, b.cols) < dispatch.level3) {
            return portable.symm(alpha, a, b, beta, c, lower, right)
        }
        val m = a.rows
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        requireShape((if (right) b.cols else b.rows) == m) {
            "symm: B is ${b.rows}x${b.cols}, expected dimension $m on the ${if (right) "cols" else "rows"} side"
        }
        if (alpha == 0.0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (c.rows == 0 || c.cols == 0) return
        f.dsymm(
            COL_MAJOR, sideOf(right), uploOf(lower), c.rows, c.cols, alpha,
            a.data, m, b.data, b.rows, beta, c.data, c.rows,
        )
    }
}
