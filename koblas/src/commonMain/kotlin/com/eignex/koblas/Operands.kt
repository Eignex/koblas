@file:Suppress("TooManyFunctions", "LongParameterList") // one check per BLAS routine, with that routine's operands

package com.eignex.koblas

/*
 * What every BLAS call requires of its operands' extents, stated once.
 *
 * Neutral on purpose: the convenience extensions, the portable dense and sparse schedules, the host
 * composition and both vendor bindings call these rather than restating them, so every layer is held to the
 * same contract by construction rather than by review. A caller who reaches a binding directly, which the
 * benchmark module does, is checked exactly as one going through [com.eignex.koblas.dense.DenseBlas] is.
 *
 * The operand is a [Matrix] because only its extents are read here, so a CSC operand answers the same
 * question a dense one does. Vectors arrive as sizes: a caller holding plain arrays has no vector to check
 * and would otherwise allocate a wrapper for each one on every call.
 *
 * Whether a declared [com.eignex.koblas.dense.MatrixStructure] suits the routine is a separate question,
 * asked by the bindings that take one from their caller; see `requireStructured` beside that enum.
 *
 * A failure raises [DimensionMismatch] through [requireShape], which is what the public surface documents.
 */

/** Two vector operands the call reads in step. */
internal fun requireSameSize(a: Int, b: Int, what: String) {
    requireShape(a == b) { "$what: sizes differ, $a and $b" }
}

/** Requires a square operand, naming [what] in a failure. */
internal fun requireSquare(a: Matrix, what: String) {
    requireShape(a.rows == a.cols) { "$what requires a square matrix, got ${a.rows}x${a.cols}" }
}

/**
 * Two operands of one shape, the extents given rather than the first operand.
 *
 * A caller that transposes one side knows its own oriented extents and has no operand in that orientation to
 * pass, which is the whole of what this compares.
 */
internal fun requireSameShape(rows: Int, cols: Int, b: Matrix, what: String) {
    requireShape(rows == b.rows && cols == b.cols) {
        "$what: ${rows}x$cols and ${b.rows}x${b.cols}, which must match"
    }
}

/** `y = alpha · op(A) · x + beta · y`: the flag decides which dimension each vector answers to. */
internal fun requireGemvOperands(a: Matrix, transposeA: Boolean, xSize: Int, ySize: Int) {
    val expectedX = if (transposeA) a.rows else a.cols
    val expectedY = if (transposeA) a.cols else a.rows
    requireShape(xSize == expectedX && ySize == expectedY) {
        "gemv: ${a.rows}x${a.cols} with transpose=$transposeA needs x=$expectedX and y=$expectedY, " +
            "got x=$xSize and y=$ySize"
    }
}

/** `y = alpha · A · x + beta · y` for a symmetric `A`, whose order both vectors answer to. */
internal fun requireSymvOperands(a: Matrix, xSize: Int, ySize: Int) {
    requireSquare(a, "symv")
    requireShape(xSize == a.rows && ySize == a.rows) {
        "symv: order ${a.rows} needs x and y of that size, got x=$xSize and y=$ySize"
    }
}

/** `A = alpha · x · yᵀ + A`, whose vectors span the destination's two dimensions. */
internal fun requireGerOperands(xSize: Int, ySize: Int, a: Matrix) {
    requireShape(xSize == a.rows && ySize == a.cols) {
        "ger: ${a.rows}x${a.cols} needs x=${a.rows} and y=${a.cols}, got x=$xSize and y=$ySize"
    }
}

/**
 * A symmetric rank update over one or two vectors of the destination's order.
 *
 * Two entries rather than one taking a variable number, since every caller knows which of the two it is and
 * a variable-length one allocates its array on each call.
 */
internal fun requireSyrOperands(a: Matrix, xSize: Int, what: String) {
    requireSquare(a, what)
    requireOrder(a, xSize, what)
}

/** [requireSyrOperands] for the two-vector update. */
internal fun requireSyr2Operands(a: Matrix, xSize: Int, ySize: Int, what: String) {
    requireSquare(a, what)
    requireOrder(a, xSize, what)
    requireOrder(a, ySize, what)
}

private fun requireOrder(a: Matrix, size: Int, what: String) {
    requireShape(size == a.rows) { "$what: order ${a.rows} needs a vector of that size, got $size" }
}

/** `op(T) · x = b` and `x = op(T) · x`, which solve or multiply in place over one vector. */
internal fun requireTriangularVectorOperands(a: Matrix, xSize: Int, what: String) {
    requireSquare(a, what)
    requireShape(xSize == a.rows) { "$what: triangle order ${a.rows} does not match operand size $xSize" }
}

/** The shared dimension of `op(A) · op(B)`, which is all a product into fresh storage constrains. */
internal fun requireProductOperands(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean, what: String) {
    val k = if (transposeA) a.rows else a.cols
    val kb = if (transposeB) b.cols else b.rows
    requireShape(k == kb) { "$what: inner dimensions differ, $k vs $kb" }
}

/** `C = alpha · op(A) · op(B) + beta · C`, with `op(A): m×k`, `op(B): k×n` and `C: m×n`. */
internal fun requireGemmOperands(
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    c: Matrix,
    what: String = "gemm",
) {
    requireProductOperands(a, transposeA, b, transposeB, what)
    val m = if (transposeA) a.cols else a.rows
    val n = if (transposeB) b.rows else b.cols
    requireShape(c.rows == m && c.cols == n) { "$what: destination must be ${m}x$n, got ${c.rows}x${c.cols}" }
}

/** `C = alpha · op(A) · op(B) + beta · C` in one triangle, so the destination is square. */
internal fun requireGemmtOperands(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean, c: Matrix) {
    requireSquare(c, "gemmt")
    requireGemmOperands(a, transposeA, b, transposeB, c, "gemmt")
}

/** `C = alpha · A · B + beta · C` for a symmetric `A`, or `C = alpha · B · A + beta · C` from the right. */
internal fun requireSymmOperands(a: Matrix, b: Matrix, c: Matrix, rightSide: Boolean) {
    requireSquare(a, "symm")
    requireSameShape(b.rows, b.cols, c, "symm")
    val order = if (rightSide) c.cols else c.rows
    requireShape(a.rows == order) {
        "symm: the symmetric operand has order ${a.rows}, and from the ${if (rightSide) "right" else "left"} " +
            "it must be $order"
    }
}

/** `C = alpha · op(A) · op(A)ᵀ + beta · C`, whose destination is square of the update's order. */
internal fun requireSyrkOperands(a: Matrix, transposeA: Boolean, c: Matrix, what: String = "syrk") {
    requireSquare(c, what)
    val order = if (transposeA) a.cols else a.rows
    requireShape(c.rows == order) { "$what: destination must be ${order}x$order, got ${c.rows}x${c.cols}" }
}

/** [requireSyrkOperands] with a second operand of the first's shape. */
internal fun requireSyr2kOperands(a: Matrix, b: Matrix, transposeA: Boolean, c: Matrix) {
    requireSameShape(a.rows, a.cols, b, "syr2k")
    requireSyrkOperands(a, transposeA, c, "syr2k")
}

/** `op(T) · X = alpha · B` and its product counterpart, from either side. */
internal fun requireTriangularMatrixOperands(a: Matrix, b: Matrix, rightSide: Boolean, what: String) {
    requireSquare(a, what)
    val order = if (rightSide) b.cols else b.rows
    requireShape(a.rows == order) {
        "$what: triangle order ${a.rows} does not match the ${if (rightSide) "columns" else "rows"} of B, $order"
    }
}
