@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.RouteKind
import com.eignex.koblas.vendor.ThreadEvidence
import com.eignex.koblas.vendor.Vendor

/**
 * One call as the binding received it.
 *
 * The operands are the objects that were handed over, not copies, so a test can ask both what they described
 * and whether they were the caller's own storage or something staged in front of it.
 */
internal class BlasCall(
    val operation: BlasOperation,
    val matrices: List<DenseMatrix> = emptyList(),
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
 * A [Blas] that records what it was asked to do and computes nothing.
 *
 * Every other dense test asks whether the numbers came back right, which a symmetric fixture or a
 * self-inverting round trip can answer correctly even when a flag was translated the wrong way. This asks the
 * other question: what did Koblas actually hand the library. A `lower` that arrives as an upper triangle, a
 * `right` that arrives as a left-side call, or an operand quietly densified out of its own spacing all show up
 * here and nowhere else.
 *
 * It records at the [Blas] seam, so it pins the translation [VendorDenseBlas] performs: Koblas's booleans into
 * [MatrixStructure] and its arrays into [DenseVector]. It cannot pin what happens below that seam, because the
 * CBLAS integers, the leading dimension and the increment are produced inside the platform bindings; checking
 * those needs a real library, which is what the vendor conformance tests are for.
 *
 * Arithmetic is deliberately absent. A double that also computed would invite tests to assert results through
 * it, and those results would be this file's arithmetic rather than any library's.
 */
internal class RecordingBlas(
    override val vendor: Vendor = Vendor.OpenBlas,
    override val directlyImplemented: Set<BlasOperation> = BlasOperation.entries.toSet(),
) : Blas {
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

    override fun routeOf(
        operation: BlasOperation,
        matrices: List<DenseMatrix>,
        vectors: List<DenseVector>,
    ): CallRoute = CallRoute(operation, RouteKind.Direct, vendor, operation.entryPoint, null, null)

    override fun dot(x: DenseVector, y: DenseVector): Double {
        record(BlasCall(BlasOperation.Dot, vectors = listOf(x, y)))
        return 0.0
    }

    override fun nrm2(x: DenseVector): Double {
        record(BlasCall(BlasOperation.Nrm2, vectors = listOf(x)))
        return 0.0
    }

    override fun asum(x: DenseVector): Double {
        record(BlasCall(BlasOperation.Asum, vectors = listOf(x)))
        return 0.0
    }

    override fun iamax(x: DenseVector): Int {
        record(BlasCall(BlasOperation.Iamax, vectors = listOf(x)))
        return 0
    }

    override fun axpy(alpha: Double, x: DenseVector, y: DenseVector) =
        record(BlasCall(BlasOperation.Axpy, vectors = listOf(x, y), scalars = listOf(alpha)))

    override fun scal(alpha: Double, x: DenseVector) =
        record(BlasCall(BlasOperation.Scal, vectors = listOf(x), scalars = listOf(alpha)))

    override fun copy(x: DenseVector, y: DenseVector) {
        record(BlasCall(BlasOperation.Copy, vectors = listOf(x, y)))
    }

    override fun swap(x: DenseVector, y: DenseVector) {
        record(BlasCall(BlasOperation.Swap, vectors = listOf(x, y)))
    }

    override fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) =
        record(BlasCall(BlasOperation.Rot, vectors = listOf(x, y), scalars = listOf(c, s)))

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens {
        record(BlasCall(BlasOperation.Rotmg, scalars = listOf(d1, d2, x1, y1)))
        return ModifiedGivens(d1, d2, x1, flag = -2.0, h11 = 0.0, h21 = 0.0, h12 = 0.0, h22 = 0.0)
    }

    override fun rotm(x: DenseVector, y: DenseVector, transformation: ModifiedGivens) =
        record(BlasCall(BlasOperation.Rotm, vectors = listOf(x, y), scalars = listOf(transformation.flag)))

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(
        BlasCall(
            BlasOperation.Gemv,
            matrices = listOf(a),
            vectors = listOf(x, y),
            scalars = listOf(alpha, beta),
            transposeA = transposeA,
        ),
    )

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) = record(
        BlasCall(
            BlasOperation.Symv,
            matrices = listOf(a),
            vectors = listOf(x, y),
            scalars = listOf(alpha, beta),
            structure = structure,
        ),
    )

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix) =
        record(BlasCall(BlasOperation.Ger, matrices = listOf(a), vectors = listOf(x, y), scalars = listOf(alpha)))

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure) = record(
        BlasCall(
            BlasOperation.Syr,
            matrices = listOf(a),
            vectors = listOf(x),
            scalars = listOf(alpha),
            structure = structure,
        ),
    )

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, structure: MatrixStructure) =
        record(
            BlasCall(
                BlasOperation.Syr2,
                matrices = listOf(a),
                vectors = listOf(x, y),
                scalars = listOf(alpha),
                structure = structure,
            ),
        )

    override fun trsv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) = record(
        BlasCall(
            BlasOperation.Trsv,
            matrices = listOf(a),
            vectors = listOf(x),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun trmv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) = record(
        BlasCall(
            BlasOperation.Trmv,
            matrices = listOf(a),
            vectors = listOf(x),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) = record(
        BlasCall(
            BlasOperation.Gemm,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            transposeA = transposeA,
            transposeB = transposeB,
        ),
    )

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            BlasOperation.Symm,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            rightSide = rightSide,
        ),
    )

    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            BlasOperation.Syrk,
            matrices = listOf(a, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            BlasOperation.Syr2k,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
        ),
    )

    override fun trmm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            BlasOperation.Trmm,
            matrices = listOf(a, b),
            scalars = listOf(alpha),
            structure = structure,
            transposeA = transposeA,
            rightSide = rightSide,
        ),
    )

    override fun trsm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) = record(
        BlasCall(
            BlasOperation.Trsm,
            matrices = listOf(a, b),
            scalars = listOf(alpha),
            structure = structure,
            transposeA = transposeA,
            rightSide = rightSide,
        ),
    )

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) = record(
        BlasCall(
            BlasOperation.Gemmt,
            matrices = listOf(a, b, c),
            scalars = listOf(alpha, beta),
            structure = structure,
            transposeA = transposeA,
            transposeB = transposeB,
        ),
    )
}
