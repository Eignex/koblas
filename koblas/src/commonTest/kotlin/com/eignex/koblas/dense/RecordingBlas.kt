@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.ThreadEvidence
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.operands.*

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
) : EngineKernels {
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
    // Whole routines run inside the shared preparation, as a vendor binding's do, so what is recorded is what a
    // library would receive: the caller's storage, or a staged copy where an input shares the destination.

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean,
        scratch: Scratch?,
    ) = prepareGemv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch?.arrays) {
            sa,
            _,
            _,
            _,
        ->
        record(EngineImplementation.Gemv, sa)
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareSymv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch?.arrays) {
            sa,
            _,
            _,
            _,
        ->
        record(EngineImplementation.Symv, sa)
    }

    override fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        scratch: Scratch?,
    ) = prepareGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch?.arrays) { _, _, _, _, _, _ ->
        record(EngineImplementation.Ger, a)
    }

    override fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareSyr(alpha, x, xOffset, xStride, a, scratch?.arrays) { _, _, _ -> record(EngineImplementation.Syr, a) }

    override fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareSyr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch?.arrays) { _, _, _, _, _, _ ->
        record(EngineImplementation.Syr2, a)
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) = prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch?.arrays) { sa ->
        record(EngineImplementation.Trsv, sa)
    }

    override fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) = prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch?.arrays) { sa ->
        record(EngineImplementation.Trmv, sa)
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch?,
    ) = prepareGemm(alpha, a, transposeA, b, transposeB, beta, c, scratch?.arrays) { sa, sb, _ ->
        record(EngineImplementation.Gemm, sa, sb, c)
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        scratch: Scratch?,
    ) = prepareSymm(this, alpha, a, b, beta, c, lower, right, scratch?.arrays) { sa, sb ->
        record(EngineImplementation.Symm, sa, sb, c)
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareSyrk(
        alpha,
        a,
        transpose,
        beta,
        c,
        lower,
        scratch?.arrays,
    ) { sa -> record(EngineImplementation.Syrk, sa, c) }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareSyr2k(alpha, a, b, transpose, beta, c, lower, scratch?.arrays) { sa, sb ->
        record(EngineImplementation.Syr2k, sa, sb, c)
    }

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) = prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch?.arrays) { sa ->
        record(EngineImplementation.Trsm, sa, b)
    }

    override fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) = prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch?.arrays) { sa ->
        record(EngineImplementation.Trmm, sa, b)
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) = prepareGemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch?.arrays) { sa, sb ->
        record(EngineImplementation.Gemmt, sa, sb, c)
    }
}
