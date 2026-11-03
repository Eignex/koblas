@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.ThreadEvidence
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.sparse.*

/** Records operand identities so tests can distinguish caller storage from staged copies. */
internal class BlasCall(
    val operation: EngineImplementation,
    val matrices: List<DenseMatrixRegion> = emptyList(),
    val vectors: List<DenseVector> = emptyList(),
    val scalars: List<Double> = emptyList(),
    val structure: MatrixStructure? = null,
    val transposeA: Boolean? = null,
    val transposeB: Boolean? = null,
    val rightSide: Boolean? = null,
) {
    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        structure?.let { append(" ").append(it) }
        transposeA?.let { append(" transA=").append(it) }
        transposeB?.let { append(" transB=").append(it) }
        rightSide?.let { append(" right=").append(it) }
        for (m in matrices) append(" ").append(m.rows).append('x').append(m.cols)
        for (v in vectors) append(" [${v.size}@${v.offset}+${v.stride}]")
    }
}

/**
 * Records [KoblasEngine] calls without computing, exposing flag and storage translation errors that
 * symmetric fixtures or round trips can conceal.
 *
 * This covers [KoblasEngine] translation into [MatrixStructure] and [DenseVector]. Platform
 * vendor tests cover the CBLAS integers, leading dimensions and increments produced below this seam.
 */
internal open class RecordingBlas(
    override val vendor: Vendor = Vendor.OpenBlas,
    override val implementations: Set<EngineImplementation> = EngineImplementation.entries.filter {
        it.entryPoint.isNotEmpty()
    }.toSet(),
) : KoblasEngine {
    override val name: String get() = vendor.vendorName.lowercase()
    private val recorded = mutableListOf<BlasCall>()

    /** Every call received so far, in order. */
    val calls: List<BlasCall> get() = recorded

    /** The one call received, failing when the count is anything else. */
    fun single(): BlasCall = recorded.single()

    private fun record(call: BlasCall) {
        recorded += call
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
        record(
            BlasCall(
                EngineImplementation.Dot,
                vectors = listOf(StridedVector(a, aOff, len, aStride), StridedVector(b, bOff, len, bStride)),
            ),
        )
        return 0.0
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        record(BlasCall(EngineImplementation.Nrm2, vectors = listOf(StridedVector(v, vOff, len, vStride))))
        return 0.0
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        record(BlasCall(EngineImplementation.Asum, vectors = listOf(StridedVector(v, vOff, len, vStride))))
        return 0.0
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        record(BlasCall(EngineImplementation.Iamax, vectors = listOf(StridedVector(v, vOff, len, vStride))))
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
    ) = record(
        BlasCall(
            EngineImplementation.Axpy,
            vectors = listOf(StridedVector(x, xOff, len, xStride), StridedVector(y, yOff, len, yStride)),
            scalars = listOf(alpha),
        ),
    )

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) = record(
        BlasCall(
            EngineImplementation.Scale,
            vectors = listOf(StridedVector(v, vOff, len, vStride)),
            scalars = listOf(alpha),
        ),
    )

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        record(
            BlasCall(
                EngineImplementation.Copy,
                vectors = listOf(StridedVector(a, aOff, len, aStride), StridedVector(b, bOff, len, bStride)),
            ),
        )
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        record(
            BlasCall(
                EngineImplementation.Swap,
                vectors = listOf(StridedVector(a, aOff, len, aStride), StridedVector(b, bOff, len, bStride)),
            ),
        )
    }

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
    ) = record(
        BlasCall(
            EngineImplementation.Rot,
            vectors = listOf(StridedVector(x, xOff, len, xStride), StridedVector(y, yOff, len, yStride)),
            scalars = listOf(c, s),
        ),
    )

    override fun computeGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(
        BlasCall(
            EngineImplementation.Gemv,
            matrices = listOf(a),
            vectors = listOf(x, y),
            scalars = listOf(alpha, beta),
            transposeA = transposeA,
        ),
    )

    override fun computeSymv(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(
        BlasCall(
            EngineImplementation.Symv,
            matrices = listOf(a),
            vectors = listOf(x, y),
            scalars = listOf(alpha, beta),
            structure = structure,
        ),
    )

    override fun computeGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) = record(
        BlasCall(EngineImplementation.Ger, matrices = listOf(a), vectors = listOf(x, y), scalars = listOf(alpha)),
    )

    override fun computeSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) = record(
        BlasCall(
            EngineImplementation.Syr,
            matrices = listOf(a),
            vectors = listOf(x),
            scalars = listOf(alpha),
            structure = structure,
        ),
    )

    override fun computeSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            EngineImplementation.Syr2,
            matrices = listOf(a),
            vectors = listOf(x, y),
            scalars = listOf(alpha),
            structure = structure,
        ),
    )

    override fun computeTrsv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        record(
            BlasCall(
                EngineImplementation.Trsv,
                matrices = listOf(a),
                vectors = listOf(x),
                structure = structure,
                transposeA = transposeA,
            ),
        )

    override fun computeTrmv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        record(
            BlasCall(
                EngineImplementation.Trmv,
                matrices = listOf(a),
                vectors = listOf(x),
                structure = structure,
                transposeA = transposeA,
            ),
        )

    override fun computeGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ) = record(
        BlasCall(
            EngineImplementation.Gemm,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            transposeA = transposeA,
            transposeB = transposeB,
        ),
    )

    override fun computeSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            EngineImplementation.Symm,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            rightSide = rightSide,
        ),
    )

    override fun computeSyrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            EngineImplementation.Syrk,
            matrices = listOf(a, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun computeSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            EngineImplementation.Syr2k,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun computeTrmm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            EngineImplementation.Trmm,
            matrices = listOf(a, b),
            scalars = listOf(alpha),
            structure = structure,
            transposeA = transposeA,
            rightSide = rightSide,
        ),
    )

    override fun computeTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            EngineImplementation.Trsm,
            matrices = listOf(a, b),
            scalars = listOf(alpha),
            structure = structure,
            transposeA = transposeA,
            rightSide = rightSide,
        ),
    )

    override fun computeGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            EngineImplementation.Gemmt,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
            transposeB = transposeB,
        ),
    )
}
