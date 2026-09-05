@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.cblas.Cblas.diagOf
import com.eignex.koblas.dense.host.cblas.Cblas.sideOf
import com.eignex.koblas.dense.host.cblas.Cblas.transOf
import com.eignex.koblas.dense.host.cblas.Cblas.uploOf

/**
 * The dense matrix routines a host CBLAS provides, over whichever [CblasCalls] the platform supplies. Both
 * host bindings are this class plus their own FFI mechanism.
 */
@Suppress("TooManyFunctions") // the BLAS surface a host library covers
public abstract class F64BlasAdapter internal constructor(
    private val f: CblasCalls,
    private val portable: F64ReferenceBlas = F64ReferenceBlas(),
    private val metadata: BackendMetadata = BackendMetadata(integerAbi = "LP64"),
) : F64Blas,
    F64RoutingBackend,
    BackendMetadataProvider {

    /** A binding that calls out, whatever the portable instance it falls back to reports. */
    override val isPortable: Boolean get() = false

    override val backendMetadata: BackendMetadata get() = metadata

    override fun route(query: F64RouteQuery): BackendRoute? = when (query) {
        is F64RouteQuery.DenseGemv, is F64RouteQuery.DenseGemm -> nativeRoute(
            query,
            this,
            fallbackWhenUnavailable = false,
        )

        else -> null
    }

    // The routines below have no host binding, so they run the portable versions. Forwarded explicitly
    // rather than by class delegation, which would route a caller's convenience overloads to the portable
    // routine instead of the accelerated one, since a delegated member calls back into the delegate.
    // A transpose is `omatcopy`, a BLAS-like extension rather than part of CBLAS. OpenBLAS does export
    // cblas_domatcopy, and binding it was measured against the portable loop on the JVM: it takes about a
    // third off a 16 or a 64, which is a matter of microseconds, and from 256 up the two sit inside the
    // run-to-run spread whichever way the shape runs, since a transpose that size is bound by memory rather
    // than by the loop. The advantage therefore shrinks with size instead of growing, leaving no dimension
    // from which the call starts paying.
    override fun transpose(a: F64DenseMatrix): F64DenseMatrix = portable.transpose(a)

    override fun syr(alpha: Double, x: F64VectorLike, a: F64DenseMatrix, lower: Boolean) {
        if (x !is F64DenseVector) return portable.syr(alpha, x, a, lower)
        requireShape(a.rows == a.cols && x.size == a.rows) { "syr shape mismatch" }
        if (alpha != 0.0 && a.rows != 0) f.dsyr(COL_MAJOR, uploOf(lower), a.rows, alpha, x.data, 1, a.data, a.rows)
    }

    override fun syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, a: F64DenseMatrix, lower: Boolean) {
        if (x !is F64DenseVector || y !is F64DenseVector) return portable.syr2(alpha, x, y, a, lower)
        requireShape(a.rows == a.cols && x.size == a.rows && y.size == a.rows) { "syr2 shape mismatch" }
        if (alpha != 0.0 && a.rows != 0) {
            f.dsyr2(
                COL_MAJOR,
                uploOf(
                    lower,
                ),
                a.rows, alpha, x.data, 1, y.data, 1, a.data, a.rows,
            )
        }
    }

    @Suppress("LongParameterList", "ReturnCount") // dsyr2k's arguments plus scratch; guard-clause style
    override fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(b.rows == a.rows && b.cols == a.cols) {
            "syr2k: B is ${b.rows}x${b.cols}, expected ${a.rows}x${a.cols} to match A"
        }
        requireShape(c.rows == n && c.cols == n) { "syr2k: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (alpha == 0.0 || k == 0) {
            scaleTriangle(kernels, c.data, n, beta, lower)
            return
        }
        if (n == 0) return
        f.dsyr2k(
            COL_MAJOR, uploOf(lower), transOf(transpose), n, k, alpha,
            a.data, a.rows, b.data, b.rows, beta, c.data, n,
        )
    }

    /** `v = beta * v`, honoring the BLAS convention that `beta == 0` overwrites without reading. */
    private fun scaleInPlace(v: DoubleArray, beta: Double) {
        when {
            beta == 0.0 -> v.fill(0.0)
            beta != 1.0 && v.isNotEmpty() -> f.dscal(v.size, beta, v, 1)
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix) {
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        f.dger(COL_MAJOR, a.rows, a.cols, alpha, x, 1, y, 1, a.data, a.rows)
    }

    override fun trsv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        nativeTriangularVector(a, x, lower, transpose, unitDiag, solve = true)
    }

    override fun trmv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
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
        workspace: Workspace?,
    ) {
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
        workspace: Workspace?,
    ) {
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
        requireTriangularMatrixShape(a, b, right, what)
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
        workspace: Workspace?,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        if (a.rows == 0 || a.cols == 0) return
        if (alpha == 0.0) {
            scaleInPlace(y, beta)
            return
        }
        f.dgemv(COL_MAJOR, transOf(transpose), a.rows, a.cols, alpha, a.data, a.rows, x, 1, beta, y, 1)
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: F64StridedMatrixView,
        x: F64StridedVectorView,
        beta: Double,
        y: F64StridedVectorView,
        transpose: Boolean,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        require(!y.overlaps(x) && !a.overlaps(y)) { "gemv: destination overlaps an input view" }
        if (x.stride < 0 || y.stride < 0) {
            return super<F64Blas>.gemv(alpha, a, x, beta, y, transpose)
        }
        if (a.rows == 0 || a.cols == 0) return
        if (alpha == 0.0) {
            scaleInPlace(y, beta)
            return
        }
        f.dgemv(
            COL_MAJOR,
            transOf(transpose),
            a.rows,
            a.cols,
            alpha,
            a.data,
            a.offset,
            a.leadingDimension,
            x.data,
            x.offset,
            x.stride,
            beta,
            y.data,
            y.offset,
            y.stride,
        )
    }

    override fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    ) {
        val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
        if (alpha == 0.0 || k == 0) {
            scaleInPlace(c.data, beta)
            return
        }
        if (m == 0 || n == 0) return
        f.dgemm(
            COL_MAJOR, transOf(transposeA), transOf(transposeB), m, n, k, alpha,
            a.data, a.rows, b.data, b.rows, beta, c.data, c.rows,
        )
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature
    override fun gemm(
        alpha: Double,
        a: F64StridedMatrixView,
        transposeA: Boolean,
        b: F64StridedMatrixView,
        transposeB: Boolean,
        beta: Double,
        c: F64StridedMatrixView,
    ) {
        val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
        require(!c.overlaps(a) && !c.overlaps(b)) { "gemm: destination overlaps an input view" }
        if (alpha == 0.0 || k == 0) {
            scaleInPlace(c, beta)
            return
        }
        if (m == 0 || n == 0) return
        f.dgemm(
            COL_MAJOR,
            transOf(transposeA),
            transOf(transposeB),
            m,
            n,
            k,
            alpha,
            a.data,
            a.offset,
            a.leadingDimension,
            b.data,
            b.offset,
            b.leadingDimension,
            beta,
            c.data,
            c.offset,
            c.leadingDimension,
        )
    }

    private fun scaleInPlace(view: F64StridedVectorView, beta: Double) {
        for (i in 0 until view.size) {
            view[i] = when (beta) {
                0.0 -> 0.0
                1.0 -> view[i]
                else -> beta * view[i]
            }
        }
    }

    private fun scaleInPlace(view: F64StridedMatrixView, beta: Double) {
        for (j in 0 until view.cols) {
            for (i in 0 until view.rows) {
                view[i, j] = when (beta) {
                    0.0 -> 0.0
                    1.0 -> view[i, j]
                    else -> beta * view[i, j]
                }
            }
        }
    }

    @Suppress("LongParameterList", "ReturnCount") // dsyrk's arguments plus scratch; guard-clause style
    override fun syrk(
        alpha: Double,
        a: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val (n, k) = requireSyrkShape(a, transpose, c, "syrk")
        if (alpha == 0.0 || k == 0) {
            scaleTriangle(kernels, c.data, n, beta, lower)
            return
        }
        if (n == 0) return
        val trans = transOf(transpose)
        f.dsyrk(COL_MAJOR, uploOf(lower), trans, n, k, alpha, a.data, a.rows, beta, c.data, n)
    }

    /**
     * The shape is checked ahead of the gate: `dsymv` takes one dimension and a leading dimension, so a
     * non-square matrix would have it read `n²` entries from a shorter array, past the end of the buffer.
     */
    override fun symv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        val n = requireSymvShape(a, x.size, y.size)
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
        workspace: Workspace?,
    ) {
        val m = requireSymmShape(a, b, c, right)
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
