@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.scaleUplo
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

/** Constructible whenever the host has OpenBLAS, independently of LAPACKE. */
internal class CblasBlas(private val f: CblasFunctions) : Blas {
    override val name: String get() = "cblas"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

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
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        // The BLAS quick-return paths do not honor the beta == 0 overwrite when a dimension is zero.
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
                        COL_MAJOR, trans, a.rows, a.cols, alpha,
                        ap.addressOf(0), a.rows, xp.addressOf(0), 1, beta, yp.addressOf(0), 1,
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
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
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
                        COL_MAJOR, transA, transB, m, n, k, alpha,
                        ap.addressOf(0), a.rows, bp.addressOf(0), b.rows, beta, cp.addressOf(0), c.rows,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList", "ReturnCount") // dsyrk's arguments plus scratch; guard-clause style
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (alpha == 0.0 || k == 0) {
            scaleUplo(vectorKernels, c.data, n, beta, uplo)
            return
        }
        if (n == 0) return
        val trans = if (transpose) TRANS else NO_TRANS
        if (uplo != Uplo.FULL) {
            val u = if (uplo == Uplo.LOWER) LOWER else UPPER
            a.data.usePinned { ap ->
                c.data.usePinned { cp ->
                    f.dsyrk(COL_MAJOR, u, trans, n, k, alpha, ap.addressOf(0), a.rows, beta, cp.addressOf(0), n)
                }
            }
            return
        }
        // dsyrk writes one triangle only, while Uplo.FULL promises a symmetric alpha term over all of C.
        val w = workspace?.take(n * n) ?: DoubleArray(n * n)
        a.data.usePinned { ap ->
            w.usePinned { wp ->
                f.dsyrk(COL_MAJOR, LOWER, trans, n, k, alpha, ap.addressOf(0), a.rows, 0.0, wp.addressOf(0), n)
            }
        }
        for (j in 0 until n) {
            for (i in j + 1 until n) w[j + i * n] = w[i + j * n]
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
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
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
                        COL_MAJOR, uplo, n, alpha,
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
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
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
        val uplo = if (lower) LOWER else UPPER
        val side = if (right) RIGHT else LEFT
        a.data.usePinned { ap ->
            b.data.usePinned { bp ->
                c.data.usePinned { cp ->
                    f.dsymm(
                        COL_MAJOR, side, uplo, c.rows, c.cols, alpha,
                        ap.addressOf(0), m, bp.addressOf(0), b.rows, beta, cp.addressOf(0), c.rows,
                    )
                }
            }
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                a.data.usePinned { ap ->
                    f.dger(
                        COL_MAJOR, a.rows, a.cols, alpha,
                        xp.addressOf(0), 1, yp.addressOf(0), 1, ap.addressOf(0), a.rows,
                    )
                }
            }
        }
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireShape(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        if (a.rows == 0) return
        a.data.usePinned { ap ->
            x.usePinned { xp ->
                f.dtrsv(
                    COL_MAJOR, uploOf(lower), transOf(transpose), diagOf(unitDiag), a.rows,
                    ap.addressOf(0), a.rows, xp.addressOf(0), 1,
                )
            }
        }
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireShape(a.rows == a.cols) { "trmv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        if (a.rows == 0) return
        a.data.usePinned { ap ->
            x.usePinned { xp ->
                f.dtrmv(
                    COL_MAJOR, uploOf(lower), transOf(transpose), diagOf(unitDiag), a.rows,
                    ap.addressOf(0), a.rows, xp.addressOf(0), 1,
                )
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
    ) = triangularMultiply(a, b, lower, transpose, unitDiag, right, solve = true)

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
    ) = triangularMultiply(a, b, lower, transpose, unitDiag, right, solve = false)

    /** dtrsm and dtrmm share every argument but the entry point, and koblas fixes alpha at 1. */
    @Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
    private fun triangularMultiply(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        solve: Boolean,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireShape(a.rows == a.cols) { "$what requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            requireShape(b.cols == a.rows) { "$what right: B has ${b.cols} cols, expected ${a.rows}" }
        } else {
            requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
        }
        if (a.rows == 0 || b.rows == 0 || b.cols == 0) return
        val side = if (right) RIGHT else LEFT
        a.data.usePinned { ap ->
            b.data.usePinned { bp ->
                val args = arrayOf(ap.addressOf(0), bp.addressOf(0))
                if (solve) {
                    f.dtrsm(
                        COL_MAJOR, side, uploOf(lower), transOf(transpose), diagOf(unitDiag),
                        b.rows, b.cols, 1.0, args[0], a.rows, args[1], b.rows,
                    )
                } else {
                    f.dtrmm(
                        COL_MAJOR, side, uploOf(lower), transOf(transpose), diagOf(unitDiag),
                        b.rows, b.cols, 1.0, args[0], a.rows, args[1], b.rows,
                    )
                }
            }
        }
    }

    private fun uploOf(lower: Boolean) = if (lower) LOWER else UPPER
    private fun transOf(transpose: Boolean) = if (transpose) TRANS else NO_TRANS
    private fun diagOf(unitDiag: Boolean) = if (unitDiag) UNIT else NON_UNIT

    /** `v = beta * v` honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)
            beta != 1.0 && v.isNotEmpty() -> v.usePinned { vp -> f.dscal(v.size, beta, vp.addressOf(0), 1) }
        }
    }
}
