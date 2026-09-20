package com.eignex.koblas

/** Requires a square operand, naming [what] in a failure. */
internal fun requireSquare(a: Matrix, what: String) {
    requireShape(a.rows == a.cols) { "$what requires a square matrix; got ${a.rows}x${a.cols}" }
}

/** The conformance two vector operands need, reported in the order the caller compared them. */
internal fun requireSameSize(a: Int, b: Int) {
    requireShape(a == b) { "size mismatch: $a vs $b" }
}

/** Checks the input and output lengths for the requested orientation. */
internal fun requireGemvShape(rows: Int, cols: Int, transpose: Boolean, x: Int, y: Int) {
    val inputs = if (transpose) rows else cols
    val outputs = if (transpose) cols else rows
    requireShape(x == inputs) { "gemv: x length $x != $inputs" }
    requireShape(y == outputs) { "gemv: y length $y != $outputs" }
}

/** The same check for a caller holding the operand rather than its extents. */
internal fun requireGemvShape(a: Matrix, transpose: Boolean, x: Int, y: Int): Unit =
    requireGemvShape(a.rows, a.cols, transpose, x, y)

/** Checks the shared dimension and destination shape of `op(A) · op(B)`. */
internal fun requireGemmShape(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean, c: Matrix): Unit =
    requireGemmShape(a.rows, a.cols, transposeA, b, transposeB, c)

/** Checks the shared dimension of a product with a fresh sparse destination. */
internal fun requireSparseProductShape(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean) {
    val aRows = if (transposeA) a.cols else a.rows
    val aCols = if (transposeA) a.rows else a.cols
    val bRows = if (transposeB) b.cols else b.rows
    val bCols = if (transposeB) b.rows else b.cols
    requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
}

/** Product shape checks using the first operand's extents. */
@Suppress("LongParameterList") // the first operand's extents in place of the operand itself
internal fun requireGemmShape(
    aRows: Int,
    aCols: Int,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    c: Matrix,
) {
    val m = if (transposeA) aCols else aRows
    val k = if (transposeA) aRows else aCols
    val kB = if (transposeB) b.cols else b.rows
    val n = if (transposeB) b.rows else b.cols
    requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
    requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
}

/** Checks the symmetric matrix of a `syr` against its vector, returning its dimension. */
internal fun requireSyrShape(a: Matrix, x: Int, what: String): Int {
    requireSquare(a, what)
    val n = a.rows
    requireShape(x == n) { "$what: x length $x != $n" }
    return n
}

/** Checks the symmetric matrix of a `syr2` against both vectors, returning its dimension. */
internal fun requireSyr2Shape(a: Matrix, x: Int, y: Int, what: String): Int {
    requireSquare(a, what)
    val n = a.rows
    requireShape(x == n && y == n) { "$what: operand lengths $x and $y must both be $n" }
    return n
}

/** Checks the triangle and the block of a `trsm` or `trmm`, returning the triangle's dimension. */
internal fun requireTriangularMatrixShape(a: Matrix, b: DenseMatrix, right: Boolean, what: String): Int {
    requireSquare(a, what)
    if (right) {
        requireShape(b.cols == a.rows) { "$what right: B has ${b.cols} cols, expected ${a.rows}" }
    } else {
        requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
    }
    return a.rows
}
