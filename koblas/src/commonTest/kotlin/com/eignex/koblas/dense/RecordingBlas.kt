@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.host.HostBlas
import com.eignex.koblas.internal.operations.MatrixStructure

/** The routine that reached the host and the matrices it was handed, so tests can tell caller storage from copies. */
internal class BlasCall(val operation: EngineImplementation, val matrices: List<DenseMatrixRegion> = emptyList()) {
    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        for (m in matrices) append(" ").append(m.rows).append('x').append(m.cols)
    }
}

/**
 * Records [KoblasEngine] host calls without computing, so a test can tell which routines a composition sent
 * across and whether the matrices they received are the caller's storage or staged copies.
 *
 * Flag and increment translation is below this seam; the vendor tests cover it against an installed library.
 */
internal open class RecordingBlas(
    override val implementations: Set<EngineImplementation> = EngineImplementation.entries.filter {
        it.entryPoint.isNotEmpty()
    }.toSet(),
) : HostBlas() {
    override val vendor: Vendor get() = Vendor.OpenBlas
    override val name: String get() = vendor.vendorName.lowercase()
    private val recorded = mutableListOf<BlasCall>()

    /** Every call received so far, in order. */
    val calls: List<BlasCall> get() = recorded

    /** The one call received, failing when the count is anything else. */
    fun single(): BlasCall = recorded.single()

    private fun record(operation: EngineImplementation, vararg matrices: DenseMatrixRegion) {
        recorded += BlasCall(operation, matrices.toList())
    }

    override val libraryPath: String get() = "/recording"
    override val version: String get() = "recording"
    override val threadEvidence: ThreadEvidence get() = ThreadEvidence.Confirmed

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double {
        record(EngineImplementation.Dot)
        return 0.0
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        record(EngineImplementation.Nrm2)
        return 0.0
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        record(EngineImplementation.Asum)
        return 0.0
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        record(EngineImplementation.Iamax)
        return 0
    }

    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = record(EngineImplementation.Axpy)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        record(EngineImplementation.Scale)

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        record(EngineImplementation.Copy)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        record(EngineImplementation.Swap)

    override fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int,
        yStride: Int,
    ) = record(EngineImplementation.Rot)
    // The shared host base checks and stages each routine, so what is recorded is what a library would receive:
    // the caller's storage, or a staged copy where an input shares the destination.

    override fun cblasGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(EngineImplementation.Gemv, a)

    override fun cblasSymv(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(EngineImplementation.Symv, a)

    override fun cblasGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) =
        record(EngineImplementation.Ger, a)

    override fun cblasSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) =
        record(EngineImplementation.Syr, a)

    override fun cblasSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(EngineImplementation.Syr2, a)

    override fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ) = record(operation, a)

    override fun cblasGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ) = record(EngineImplementation.Gemm, a, b, c)

    override fun cblasGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(EngineImplementation.Gemmt, a, b, c)

    override fun cblasSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ) = record(EngineImplementation.Symm, a, b, c)

    override fun cblasSyrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(EngineImplementation.Syrk, a, c)

    override fun cblasSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(EngineImplementation.Syr2k, a, b, c)

    override fun cblasTriangularMatrix(
        operation: EngineImplementation,
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = record(operation, a, b)
}
