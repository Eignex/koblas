@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.requireShape
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.MissingVendorException

/**
 * Dense Level 2 and 3 served by whole vendor BLAS calls across one boundary.
 *
 * Every public form, owning matrix or borrowed view, is validated and described the same way and then handed to
 * the same entry point, with the transpose and the stored triangle travelling beside the operand as flags.
 * Koblas contributes shape and aliasing validation and nothing else; the arithmetic, including its treatment of
 * zero multipliers, infinities and the order accumulations happen in, is the selected vendor's.
 *
 * [vendor] is null on a host where no supported library was found. Containers, Level 1 and the generic
 * primitives keep working there; every operation on this seam raises [MissingVendorException] instead, because
 * an accelerator-dependent call has nothing to fall back to and a silent portable substitute would be a
 * different implementation reported under the same name.
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
    ) {
        requireGemvShape(a.rows, a.cols, transpose, x.size, y.size)
        blas.gemv(alpha, a, transpose, x.asVector(), beta, y.asVector())
    }

    override fun transpose(a: DenseMatrix): DenseMatrix {
        // A storage transform rather than arithmetic: the standard has no entry point for it, so it stays a
        // Kotlin loop instead of borrowing a BLAS-like extension that only some vendors export.
        val result = DenseMatrix(a.cols, a.rows)
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) result.data[j + i * a.cols] = a.data[i + j * a.rows]
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
    ) {
        requireGemmShape(a.rows, a.cols, transposeA, b.rows, b.cols, transposeB, c.rows, c.cols)
        requireDistinctDestination(c.data, a.data, b.data, "gemm")
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
    ) {
        requireGemmShape(a.rows, a.cols, transposeA, b.rows, b.cols, transposeB, c.rows, c.cols)
        requireShape(c.rows == c.cols) { "gemmt: destination must be square, got ${c.rows}x${c.cols}" }
        requireDistinctDestination(c.data, a.data, b.data, "gemmt")
        blas.gemmt(alpha, a, transposeA, b, transposeB, beta, c, triangle(lower))
    }

    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        val order = if (transpose) a.cols else a.rows
        requireShape(c.rows == order && c.cols == order) {
            "syrk: destination must be ${order}x$order, got ${c.rows}x${c.cols}"
        }
        requireDistinctDestination(c.data, a.data, null, "syrk")
        blas.syrk(alpha, a, transpose, beta, c, triangle(lower))
    }

    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.cols && y.size == a.rows) { "symv: vector sizes do not match ${a.rows}x${a.cols}" }
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
    ) {
        requireShape(a.rows == a.cols) { "symm: symmetric operand must be square, got ${a.rows}x${a.cols}" }
        requireShape(b.rows == c.rows && b.cols == c.cols) { "symm: B and C shapes differ" }
        val order = if (right) c.cols else c.rows
        requireShape(a.rows == order) { "symm: symmetric operand order ${a.rows} does not match $order" }
        requireDistinctDestination(c.data, a.data, b.data, "symm")
        blas.symm(alpha, a, symmetricStructure(lower), b, beta, c, rightSide = right)
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireShape(x.size == a.rows && y.size == a.cols) { "ger: vector sizes do not match ${a.rows}x${a.cols}" }
        blas.ger(alpha, x.asVector(), y.asVector(), a)
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireShape(a.rows == a.cols) { "syr: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "syr: vector size ${x.size} does not match order ${a.rows}" }
        blas.syr(alpha, x, a, triangle(lower))
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireShape(a.rows == a.cols) { "syr2: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows && y.size == a.rows) { "syr2: vector sizes do not match order ${a.rows}" }
        blas.syr2(alpha, x, y, a, triangle(lower))
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    ) {
        val order = if (transpose) a.cols else a.rows
        requireShape(a.rows == b.rows && a.cols == b.cols) { "syr2k: A and B shapes differ" }
        requireShape(c.rows == order && c.cols == order) {
            "syr2k: destination must be ${order}x$order, got ${c.rows}x${c.cols}"
        }
        requireDistinctDestination(c.data, a.data, b.data, "syr2k")
        blas.syr2k(alpha, a, b, transpose, beta, c, triangle(lower))
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularShape(a, x.size, "trsv")
        blas.trsv(a, triangle(lower, unitDiag), transpose, x.asVector())
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularShape(a, x.size, "trmv")
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
    ) {
        requireTriangularShape(a, if (right) b.cols else b.rows, "trsm")
        requireDistinctDestination(b.data, a.data, null, "trsm")
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
    ) {
        requireTriangularShape(a, if (right) b.cols else b.rows, "trmm")
        requireDistinctDestination(b.data, a.data, null, "trmm")
        blas.trmm(alpha, a, triangle(lower, unitDiag), transpose, b, rightSide = right)
    }
}

private fun triangle(lower: Boolean, unitDiag: Boolean = false): MatrixStructure = when {
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

private fun requireGemvShape(rows: Int, cols: Int, transpose: Boolean, x: Int, y: Int) {
    val expectedX = if (transpose) rows else cols
    val expectedY = if (transpose) cols else rows
    requireShape(x == expectedX && y == expectedY) {
        "gemv: ${rows}x$cols with transpose=$transpose needs x=$expectedX and y=$expectedY, got x=$x and y=$y"
    }
}

@Suppress("LongParameterList") // both operand shapes plus the destination
private fun requireGemmShape(
    aRows: Int,
    aCols: Int,
    transposeA: Boolean,
    bRows: Int,
    bCols: Int,
    transposeB: Boolean,
    cRows: Int,
    cCols: Int,
) {
    val m = if (transposeA) aCols else aRows
    val k = if (transposeA) aRows else aCols
    val kb = if (transposeB) bCols else bRows
    val n = if (transposeB) bRows else bCols
    requireShape(k == kb) { "gemm: inner dimensions differ, $k vs $kb" }
    requireShape(cRows == m && cCols == n) { "gemm: destination must be ${m}x$n, got ${cRows}x$cCols" }
}

private fun requireTriangularShape(a: DenseMatrix, order: Int, what: String) {
    requireShape(a.rows == a.cols) { "$what: triangle must be square, got ${a.rows}x${a.cols}" }
    requireShape(a.rows == order) { "$what: triangle order ${a.rows} does not match operand order $order" }
}

/**
 * Refuses a destination that shares a buffer with an input.
 *
 * BLAS states that the destination of these operations does not overlap their inputs, and a vendor is free to
 * read an operand after writing part of the result. Rejecting here keeps that undefined case from becoming a
 * silent wrong answer, and it happens before anything is written.
 */
private fun requireDistinctDestination(c: DoubleArray, a: DoubleArray, b: DoubleArray?, what: String) {
    require(c !== a && c !== b) { "$what: destination shares a buffer with an input" }
}
