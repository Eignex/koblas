@file:Suppress("TooManyFunctions", "LongParameterList") // one check per CBLAS routine, with that routine's arguments

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.requireShape

/*
 * What every BLAS call requires of its operands, stated once.
 *
 * These used to be written out three times: in the seam above, and again in each of the two platform bindings,
 * which between them restated thirty identical lines. Nothing kept the copies in step, and they did drift: a
 * `gemv` shape check that ignored the transpose flag had to be found and fixed in both bindings separately.
 *
 * So the rule lives here and the bindings call it. Both platforms are held to the same contract by
 * construction rather than by review, and a caller who reaches a binding directly, which the benchmark module
 * and the Level 2 convenience extensions both do, is checked exactly as one going through
 * [com.eignex.koblas.dense.DenseBlas] is.
 *
 * A shape failure raises [com.eignex.koblas.DimensionMismatch] through [requireShape], which is what the public
 * surface documents; a structure or aliasing failure is an ordinary argument error.
 */

/** Two vector operands the call reads in step. */
internal fun requireSameLength(x: DenseVector, y: DenseVector, what: String) {
    requireShape(x.size == y.size) { "$what: vector sizes differ, ${x.size} and ${y.size}" }
}

/**
 * A square operand with a stored triangle and a stored diagonal.
 *
 * None of the routines that take one carry a `diag` flag, so there is no way to tell the vendor that a
 * diagonal is implied. An operand that says its diagonal is implicit is therefore rejected rather than served
 * by a call that would read or write it anyway; the triangular routines, which do carry the flag, take
 * [requireTriangular] instead.
 */
internal fun requireStructured(a: DenseMatrix, structure: MatrixStructure, what: String) {
    val stored = structure != MatrixStructure.General &&
        structure != MatrixStructure.UnitLower &&
        structure != MatrixStructure.UnitUpper
    require(stored) { "$what requires a stored triangle with a stored diagonal" }
    requireShape(a.rows == a.cols) { "$what requires a square matrix, got ${a.rows}x${a.cols}" }
}

/** A triangular operand, stored or with an implicit unit diagonal. */
internal fun requireTriangular(a: DenseMatrix, structure: MatrixStructure, what: String) {
    val triangular = structure == MatrixStructure.TriangularLower ||
        structure == MatrixStructure.TriangularUpper ||
        structure == MatrixStructure.UnitLower ||
        structure == MatrixStructure.UnitUpper
    require(triangular) { "$what requires a triangular matrix" }
    requireShape(a.rows == a.cols) { "$what requires a square matrix, got ${a.rows}x${a.cols}" }
}

/** `y = alpha · op(A) · x + beta · y`: the flag decides which dimension each vector answers to. */
internal fun requireGemvOperands(a: DenseMatrix, transposeA: Boolean, x: DenseVector, y: DenseVector) {
    val expectedX = if (transposeA) a.rows else a.cols
    val expectedY = if (transposeA) a.cols else a.rows
    requireShape(x.size == expectedX && y.size == expectedY) {
        "gemv: ${a.rows}x${a.cols} with transpose=$transposeA needs x=$expectedX and y=$expectedY, " +
            "got x=${x.size} and y=${y.size}"
    }
}

/** `y = alpha · A · x + beta · y` for a symmetric `A`. */
internal fun requireSymvOperands(a: DenseMatrix, structure: MatrixStructure, x: DenseVector, y: DenseVector) {
    requireStructured(a, structure, "symv")
    requireShape(x.size == a.cols && y.size == a.rows) {
        "symv: order ${a.rows} needs x and y of that size, got x=${x.size} and y=${y.size}"
    }
}

/** `A = alpha · x · yᵀ + A`, whose vectors span the destination's two dimensions. */
internal fun requireGerOperands(x: DenseVector, y: DenseVector, a: DenseMatrix) {
    requireShape(x.size == a.rows && y.size == a.cols) {
        "ger: ${a.rows}x${a.cols} needs x=${a.rows} and y=${a.cols}, got x=${x.size} and y=${y.size}"
    }
}

/** A symmetric rank update over one or two vectors of the destination's order. */
internal fun requireSyrOperands(
    a: DenseMatrix,
    structure: MatrixStructure,
    what: String,
    vararg vectors: DenseVector,
) {
    requireStructured(a, structure, what)
    for (x in vectors) {
        requireShape(x.size == a.rows) { "$what: order ${a.rows} needs a vector of that size, got ${x.size}" }
    }
}

/** `op(T) · x = b` and `x = op(T) · x`, which solve or multiply in place over one vector. */
internal fun requireTriangularVectorOperands(
    a: DenseMatrix,
    structure: MatrixStructure,
    x: DenseVector,
    what: String,
) {
    requireTriangular(a, structure, what)
    requireShape(x.size == a.rows) { "$what: triangle order ${a.rows} does not match operand size ${x.size}" }
}

/** `C = alpha · op(A) · op(B) + beta · C`, with `op(A): m×k`, `op(B): k×n` and `C: m×n`. */
internal fun requireGemmOperands(
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    what: String = "gemm",
) {
    val m = if (transposeA) a.cols else a.rows
    val k = if (transposeA) a.rows else a.cols
    val kb = if (transposeB) b.cols else b.rows
    val n = if (transposeB) b.rows else b.cols
    requireShape(k == kb) { "$what: inner dimensions differ, $k vs $kb" }
    requireShape(c.rows == m && c.cols == n) { "$what: destination must be ${m}x$n, got ${c.rows}x${c.cols}" }
}

/** `C = alpha · A · B + beta · C` for a symmetric `A`, or `C = alpha · B · A + beta · C` from the right. */
internal fun requireSymmOperands(
    a: DenseMatrix,
    structure: MatrixStructure,
    b: DenseMatrix,
    c: DenseMatrix,
    rightSide: Boolean,
) {
    requireStructured(a, structure, "symm")
    requireShape(b.rows == c.rows && b.cols == c.cols) {
        "symm: B is ${b.rows}x${b.cols} and C is ${c.rows}x${c.cols}, which must match"
    }
    val order = if (rightSide) c.cols else c.rows
    requireShape(a.rows == order) {
        "symm: the symmetric operand has order ${a.rows}, and from the ${if (rightSide) "right" else "left"} " +
            "it must be $order"
    }
}

/** `C = alpha · op(A) · op(A)ᵀ + beta · C`, whose destination is square of the update's order. */
internal fun requireSyrkOperands(
    a: DenseMatrix,
    transposeA: Boolean,
    c: DenseMatrix,
    structure: MatrixStructure,
    what: String = "syrk",
) {
    requireStructured(c, structure, what)
    val order = if (transposeA) a.cols else a.rows
    requireShape(c.rows == order) { "$what: destination must be ${order}x$order, got ${c.rows}x${c.cols}" }
}

/** [requireSyrkOperands] with a second operand of the first's shape. */
internal fun requireSyr2kOperands(
    a: DenseMatrix,
    b: DenseMatrix,
    transposeA: Boolean,
    c: DenseMatrix,
    structure: MatrixStructure,
) {
    requireShape(a.rows == b.rows && a.cols == b.cols) {
        "syr2k: A is ${a.rows}x${a.cols} and B is ${b.rows}x${b.cols}, which must match"
    }
    requireSyrkOperands(a, transposeA, c, structure, "syr2k")
}

/** `op(T) · X = alpha · B` and its product counterpart, from either side. */
internal fun requireTriangularMatrixOperands(
    a: DenseMatrix,
    structure: MatrixStructure,
    b: DenseMatrix,
    rightSide: Boolean,
    what: String,
) {
    requireTriangular(a, structure, what)
    val order = if (rightSide) b.cols else b.rows
    requireShape(a.rows == order) {
        "$what: triangle order ${a.rows} does not match the ${if (rightSide) "columns" else "rows"} of B, $order"
    }
}

/** `C = alpha · op(A) · op(B) + beta · C` in one triangle, so the destination is square and structured. */
internal fun requireGemmtOperands(
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    structure: MatrixStructure,
) {
    requireStructured(c, structure, "gemmt")
    requireGemmOperands(a, transposeA, b, transposeB, c, "gemmt")
}

/**
 * Refuses a destination that shares a buffer with an input.
 *
 * BLAS states that the destination of these operations does not overlap their inputs, and a vendor is free to
 * read an operand after writing part of the result. Rejecting here keeps that undefined case from becoming a
 * silent wrong answer, and it happens before anything is written.
 */
internal fun requireDistinctDestination(c: DenseMatrix, a: DenseMatrix, b: DenseMatrix?, what: String) {
    require(c.data !== a.data && c.data !== b?.data) { "$what: destination shares a buffer with an input" }
}
