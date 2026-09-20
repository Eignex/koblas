@file:Suppress("TooManyFunctions", "LongParameterList") // one check per CBLAS routine, with that routine's arguments

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.requireShape

/*
 * What every BLAS call requires of its operands, stated once.
 *
 * The seam above and both platform bindings call these rather than restating them, so the two platforms are
 * held to the same contract by construction rather than by review. A caller who reaches a binding directly,
 * which the benchmark module and the Level 2 convenience extensions both do, is checked exactly as one going
 * through [com.eignex.koblas.dense.DenseBlas] is.
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

/**
 * `y = alpha · op(A) · x + beta · y`: the flag decides which dimension each vector answers to.
 *
 * Sizes rather than operands, because a caller holding plain arrays has no vector to check and would have to
 * wrap each one to ask. The wrappers are small, and a Level 2 call that allocates two of them per invocation
 * is exactly the per-call cost the panels below it exist to avoid.
 */
internal fun requireGemvOperands(a: DenseMatrix, transposeA: Boolean, xSize: Int, ySize: Int) {
    val expectedX = if (transposeA) a.rows else a.cols
    val expectedY = if (transposeA) a.cols else a.rows
    requireShape(xSize == expectedX && ySize == expectedY) {
        "gemv: ${a.rows}x${a.cols} with transpose=$transposeA needs x=$expectedX and y=$expectedY, " +
            "got x=$xSize and y=$ySize"
    }
}

/** [requireGemvOperands] for a caller that already holds its operands as vectors. */
internal fun requireGemvOperands(a: DenseMatrix, transposeA: Boolean, x: DenseVector, y: DenseVector): Unit =
    requireGemvOperands(a, transposeA, x.size, y.size)

/** `y = alpha · A · x + beta · y` for a symmetric `A`. Sizes for the reason [requireGemvOperands] gives. */
internal fun requireSymvOperands(a: DenseMatrix, structure: MatrixStructure, xSize: Int, ySize: Int) {
    requireStructured(a, structure, "symv")
    requireShape(xSize == a.cols && ySize == a.rows) {
        "symv: order ${a.rows} needs x and y of that size, got x=$xSize and y=$ySize"
    }
}

/** [requireSymvOperands] for a caller that already holds its operands as vectors. */
internal fun requireSymvOperands(a: DenseMatrix, structure: MatrixStructure, x: DenseVector, y: DenseVector): Unit =
    requireSymvOperands(a, structure, x.size, y.size)

/** `A = alpha · x · yᵀ + A`, whose vectors span the destination's two dimensions. */
internal fun requireGerOperands(xSize: Int, ySize: Int, a: DenseMatrix) {
    requireShape(xSize == a.rows && ySize == a.cols) {
        "ger: ${a.rows}x${a.cols} needs x=${a.rows} and y=${a.cols}, got x=$xSize and y=$ySize"
    }
}

/** [requireGerOperands] for a caller that already holds its operands as vectors. */
internal fun requireGerOperands(x: DenseVector, y: DenseVector, a: DenseMatrix): Unit =
    requireGerOperands(x.size, y.size, a)

/**
 * A symmetric rank update over one or two vectors of the destination's order.
 *
 * Two entries rather than one taking a variable number, since every caller knows which of the two it is and
 * a variable-length one allocates its array on each call.
 */
internal fun requireSyrOperands(a: DenseMatrix, structure: MatrixStructure, what: String, x: DenseVector) {
    requireStructured(a, structure, what)
    requireOrder(a, x.size, what)
}

/** [requireSyrOperands] for the two-vector update. */
internal fun requireSyr2Operands(
    a: DenseMatrix,
    structure: MatrixStructure,
    what: String,
    x: DenseVector,
    y: DenseVector,
) {
    requireStructured(a, structure, what)
    requireOrder(a, x.size, what)
    requireOrder(a, y.size, what)
}

private fun requireOrder(a: DenseMatrix, size: Int, what: String) {
    requireShape(size == a.rows) { "$what: order ${a.rows} needs a vector of that size, got $size" }
}

/** `op(T) · x = b` and `x = op(T) · x`, which solve or multiply in place over one vector. */
internal fun requireTriangularVectorOperands(a: DenseMatrix, structure: MatrixStructure, xSize: Int, what: String) {
    requireTriangular(a, structure, what)
    requireShape(xSize == a.rows) { "$what: triangle order ${a.rows} does not match operand size $xSize" }
}

/** [requireTriangularVectorOperands] for a caller that already holds its operand as a vector. */
internal fun requireTriangularVectorOperands(
    a: DenseMatrix,
    structure: MatrixStructure,
    x: DenseVector,
    what: String,
): Unit = requireTriangularVectorOperands(a, structure, x.size, what)

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
