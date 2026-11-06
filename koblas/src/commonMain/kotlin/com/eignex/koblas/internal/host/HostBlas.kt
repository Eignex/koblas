@file:Suppress("LongParameterList", "TooManyFunctions") // the CBLAS double-precision Level 2 and 3 surface

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.storage.sameRegion

/**
 * A host library's Level 2 and 3 routines, checked and staged as every implementation is before its CBLAS call.
 *
 * A binding supplies the calls themselves, which receive operands already validated and free of overlap with
 * their destination. A routine the library does not export fails before any operand is touched.
 */
internal abstract class HostBlas : EngineKernels {
    protected abstract fun cblasGemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    )

    protected abstract fun cblasSymv(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    )

    protected abstract fun cblasGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix)

    protected abstract fun cblasSyr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure)

    protected abstract fun cblasSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrix,
        structure: MatrixStructure,
    )

    /** `cblas_dtrsv` or `cblas_dtrmv`, as [operation] names, which share one signature. */
    protected abstract fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    )

    protected abstract fun cblasGemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    protected abstract fun cblasGemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )

    protected abstract fun cblasSymm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        rightSide: Boolean,
    )

    protected abstract fun cblasSyrk(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )

    protected abstract fun cblasSyr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )

    /** `cblas_dtrsm` or `cblas_dtrmm`, as [operation] names, which share one signature. */
    protected abstract fun cblasTriangularMatrix(
        operation: EngineImplementation,
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    )

    private fun requireExport(operation: EngineImplementation) {
        if (operation !in implementations) {
            throw UnsupportedOperationException("${operation.entryPoint} is not exported by $name")
        }
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemv)
        val rows = if (transpose) a.cols else a.rows
        val depth = if (transpose) a.rows else a.cols
        prepareMatrixVector(this, alpha, a, transpose, x, xOffset, xStride, beta, y, yOffset, yStride, null, scratch) {
                sa,
                sx,
                so,
                ss,
            ->
            cblasGemv(
                alpha,
                sa,
                transpose,
                StridedVector(sx, so, depth, ss),
                beta,
                StridedVector(y, yOffset, rows, yStride),
            )
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Symv)
        requireSquare(a, "symv")
        prepareMatrixVector(this, alpha, a, false, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch) {
                sa,
                sx,
                so,
                ss,
            ->
            val n = a.rows
            cblasSymv(
                alpha,
                sa,
                symmetricStructure(lower),
                StridedVector(sx, so, n, ss),
                beta,
                StridedVector(y, yOffset, n, yStride),
            )
        }
    }

    override fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrix,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Ger)
        prepareRankUpdate(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch) {
                sx,
                xo,
                xs,
                sy,
                yo,
                ys,
            ->
            cblasGer(alpha, StridedVector(sx, xo, a.rows, xs), StridedVector(sy, yo, a.cols, ys), a)
        }
    }

    override fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr)
        requireSquare(a, "syr")
        prepareRankUpdate(alpha, x, xOffset, xStride, x, xOffset, xStride, a, scratch) { sx, xo, xs, _, _, _ ->
            cblasSyr(alpha, StridedVector(sx, xo, a.rows, xs), a, symmetricStructure(lower))
        }
    }

    override fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr2)
        requireSquare(a, "syr2")
        prepareRankUpdate(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch) {
                sx,
                xo,
                xs,
                sy,
                yo,
                ys,
            ->
            val n = a.rows
            cblasSyr2(alpha, StridedVector(sx, xo, n, xs), StridedVector(sy, yo, n, ys), a, symmetricStructure(lower))
        }
    }

    override fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trsv)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch) { sa ->
            val vector = StridedVector(x, xOffset, a.rows, xStride)
            cblasTriangularVector(EngineImplementation.Trsv, sa, triangle(lower, unitDiag), transpose, vector)
        }
    }

    override fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trmv)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch) { sa ->
            val vector = StridedVector(x, xOffset, a.rows, xStride)
            cblasTriangularVector(EngineImplementation.Trmv, sa, triangle(lower, unitDiag), transpose, vector)
        }
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemm)
        requireGemmOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        prepareProduct(alpha, a, b, beta, c, depth, null, null, null, scratch) { sa, sb ->
            cblasGemm(alpha, sa, transposeA, sb, transposeB, beta, c)
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemmt)
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            cblasGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, symmetricStructure(lower))
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Symm)
        requireSymmOperands(a, b, c, right)
        // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
        val aTriangle = if (sameRegion(a, b)) null else lower
        prepareProduct(alpha, a, b, beta, c, a.rows, null, aTriangle, this, scratch) { sa, sb ->
            cblasSymm(alpha, sa, symmetricStructure(lower), sb, beta, c, right)
        }
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syrk)
        requireSyrkOperands(a, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        prepareProduct(alpha, a, a, beta, c, depth, lower, null, null, scratch) { sa, _ ->
            cblasSyrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr2k)
        requireSyr2kOperands(a, b, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            cblasSyr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trsm)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch) { sa ->
            cblasTriangularMatrix(EngineImplementation.Trsm, alpha, sa, triangle(lower, unitDiag), transpose, b, right)
        }
    }

    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trmm)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch) { sa ->
            cblasTriangularMatrix(EngineImplementation.Trmm, alpha, sa, triangle(lower, unitDiag), transpose, b, right)
        }
    }
}
