@file:Suppress("LongParameterList", "TooManyFunctions") // the CBLAS double-precision Level 2 and 3 surface

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.storage.dense.*

/**
 * A host library's Level 2 and 3 routines, checked and staged as every implementation is before its CBLAS call.
 *
 * A binding supplies the calls themselves, which receive operands already validated and free of overlap with
 * their destination. A routine the library does not export fails before any operand is touched.
 */
internal abstract class HostBlas : KernelProvider {
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
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemv)
        prepareMatrixVector(this, alpha, a, transpose, x, beta, y, null, scratch) { sa, sx ->
            cblasGemv(alpha, sa, transpose, sx, beta, y)
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Symv)
        requireSquare(a, "symv")
        prepareMatrixVector(this, alpha, a, false, x, beta, y, lower, scratch) { sa, sx ->
            cblasSymv(alpha, sa, symmetricStructure(lower), sx, beta, y)
        }
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, scratch: Scratch?) {
        requireExport(EngineImplementation.Ger)
        prepareRankUpdate(alpha, x, y, a, scratch) { sx, sy -> cblasGer(alpha, sx, sy, a) }
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean, scratch: Scratch?) {
        requireExport(EngineImplementation.Syr)
        requireSquare(a, "syr")
        prepareRankUpdate(alpha, x, x, a, scratch) { sx, _ -> cblasSyr(alpha, sx, a, symmetricStructure(lower)) }
    }

    override fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrix,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr2)
        requireSquare(a, "syr2")
        prepareRankUpdate(alpha, x, y, a, scratch) { sx, sy -> cblasSyr2(alpha, sx, sy, a, symmetricStructure(lower)) }
    }

    override fun trsv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trsv)
        prepareTriangularVector(a, x, lower, unitDiag, "trsv", scratch) { sa ->
            cblasTriangularVector(EngineImplementation.Trsv, sa, triangle(lower, unitDiag), transpose, x)
        }
    }

    override fun trmv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trmv)
        prepareTriangularVector(a, x, lower, unitDiag, "trmv", scratch) { sa ->
            cblasTriangularVector(EngineImplementation.Trmv, sa, triangle(lower, unitDiag), transpose, x)
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
