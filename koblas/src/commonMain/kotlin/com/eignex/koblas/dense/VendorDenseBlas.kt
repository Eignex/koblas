@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.vendor.*

/**
 * Dense Level 2 and 3 served by whole vendor BLAS calls across one boundary.
 *
 * Koblas contributes shape and aliasing validation and nothing else; the arithmetic, including its treatment of
 * zero multipliers, infinities and the order accumulations happen in, is the selected vendor's.
 *
 * The shape rules are the ones in `Operands.kt`, which [Blas]'s implementations call as well. Checking at both
 * seams is deliberate rather than redundant: a caller reaching a binding directly, which the benchmark module
 * and the Level 2 convenience extensions both do, is owed the same answer, and checking here means a bad shape
 * is reported as a bad shape on a host with no library at all rather than as the missing library. The cost is
 * a handful of integer comparisons in front of a call that copies whole matrices across a foreign boundary.
 *
 * Aliasing is checked only here, because it is stated in terms of the buffers Koblas's own containers own.
 *
 * [vendor] is null on a host where no supported library was found. Containers, Level 1 and the generic
 * primitives keep working there; every operation on this seam raises [MissingVendorException] instead, because
 * an accelerator-dependent call has nothing to fall back to and a silent portable substitute would be a
 * different implementation reported under the same name.
 *
 * A workspace offered here is not used, because there is nothing for it to lend. A whole-call vendor route
 * hands the caller's storage to the library and rejects an overlap rather than staging around it, so it takes
 * no scratch of its own; the parameter is on the seam because the built-in implementations do take one.
 */
internal class VendorDenseBlas(private val vendor: Blas?) : DenseBlas {
    private val blas: Blas get() = vendor ?: throw MissingVendorException()

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.asVector(), y.asVector())
        blas.gemv(alpha, a, transpose, x.asVector(), beta, y.asVector())
    }

    override fun transpose(a: DenseMatrix): DenseMatrix {
        val result = DenseMatrix(a.cols, a.rows)
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) result.values[j + i * a.cols] = a.values[i + j * a.rows]
        }
        return result
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        requireDistinctDestination(c, a, b, "gemm")
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
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
        workspace: Workspace?,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c, symmetricStructure(lower))
        requireDistinctDestination(c, a, b, "gemmt")
        blas.gemmt(alpha, a, transposeA, b, transposeB, beta, c, symmetricStructure(lower))
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyrkOperands(a, transpose, c, symmetricStructure(lower))
        requireDistinctDestination(c, a, null, "syrk")
        blas.syrk(alpha, a, transpose, beta, c, symmetricStructure(lower))
    }

    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSymvOperands(a, symmetricStructure(lower), x.asVector(), y.asVector())
        blas.symv(alpha, a, symmetricStructure(lower), x.asVector(), beta, y.asVector())
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, symmetricStructure(lower), b, c, right)
        requireDistinctDestination(c, a, b, "symm")
        blas.symm(alpha, a, symmetricStructure(lower), b, beta, c, rightSide = right)
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireGerOperands(x.asVector(), y.asVector(), a)
        blas.ger(alpha, x.asVector(), y.asVector(), a)
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyrOperands(a, symmetricStructure(lower), "syr", x)
        blas.syr(alpha, x, a, symmetricStructure(lower))
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyr2Operands(a, symmetricStructure(lower), "syr2", x, y)
        blas.syr2(alpha, x, y, a, symmetricStructure(lower))
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2kOperands(a, b, transpose, c, symmetricStructure(lower))
        requireDistinctDestination(c, a, b, "syr2k")
        blas.syr2k(alpha, a, b, transpose, beta, c, symmetricStructure(lower))
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.asVector(), "trsv")
        blas.trsv(a, triangle(lower, unitDiag), transpose, x.asVector())
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.asVector(), "trmv")
        blas.trmv(a, triangle(lower, unitDiag), transpose, x.asVector())
    }

    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, triangle(lower, unitDiag), b, right, "trsm")
        requireDistinctDestination(b, a, null, "trsm")
        blas.trsm(alpha, a, triangle(lower, unitDiag), transpose, b, rightSide = right)
    }

    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, triangle(lower, unitDiag), b, right, "trmm")
        requireDistinctDestination(b, a, null, "trmm")
        blas.trmm(alpha, a, triangle(lower, unitDiag), transpose, b, rightSide = right)
    }
}

/**
 * The triangle and diagonal a genuinely triangular operand declares.
 *
 * Only the four triangular routines use this. A symmetric destination takes [symmetricStructure] instead, even
 * though both spellings reach the same CBLAS `uplo`: a rank update's destination is stored as a symmetric
 * matrix is, and calling it triangular would declare the other half to be zeros it is not.
 */
internal fun triangle(lower: Boolean, unitDiag: Boolean): MatrixStructure = when {
    unitDiag && lower -> MatrixStructure.UnitLower
    unitDiag -> MatrixStructure.UnitUpper
    lower -> MatrixStructure.TriangularLower
    else -> MatrixStructure.TriangularUpper
}

/** The stored triangle a symmetric operand declares, shared with the callers that state it themselves. */
internal fun symmetricStructure(lower: Boolean): MatrixStructure =
    if (lower) MatrixStructure.SymmetricLower else MatrixStructure.SymmetricUpper

/** A caller-owned array as the contiguous vector operand BLAS takes. */
internal fun DoubleArray.asVector(): DenseVector = DenseVector.wrap(this)
