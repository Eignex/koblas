package com.eignex.koblas

/** Operands whose shapes do not fit the routine. */
public class DimensionMismatch(message: String) : IllegalArgumentException(message)

/** A numerical outcome that prevents a requested factorization or solve from completing. */
public sealed class KoblasException(message: String) : ArithmeticException(message)

/**
 * A factorization met an exactly zero pivot, so the matrix it came from has no inverse.
 *
 * @property position the 0-based pivot index where the factorization failed.
 * @param message what failed, naming the routine and the position.
 */
public class SingularMatrix(public val position: Int, message: String) : KoblasException(message)

/**
 * A sparse Cholesky or a strict `L·D·Lᵀ` met a pivot that was zero, negative or NaN.
 *
 * @property position the 0-based column whose pivot failed.
 * @property pivot the offending diagonal value.
 * @param message what failed, naming the position, the pivot and the policy that would continue past it.
 */
public class NotPositiveDefinite(public val position: Int, public val pivot: Double, message: String) :
    KoblasException(
        message,
    )

/** `require` for a shape check, throwing [DimensionMismatch] instead of a bare `IllegalArgumentException`. */
internal inline fun requireShape(condition: Boolean, message: () -> String) {
    if (!condition) throw DimensionMismatch(message())
}

/** `require` for an index check, so every storage reports the same standard type. */
internal inline fun requireIndex(condition: Boolean, message: () -> String) {
    if (!condition) throw IndexOutOfBoundsException(message())
}

/** The shape every factorization and triangular routine needs, naming [what] so the message says which one. */
internal fun requireSquare(a: Matrix, what: String) {
    requireShape(a.rows == a.cols) { "$what requires a square matrix; got ${a.rows}x${a.cols}" }
}

/** The conformance two vector operands need, reported in the order the caller compared them. */
internal fun requireSameSize(a: Int, b: Int) {
    requireShape(a == b) { "size mismatch: $a vs $b" }
}

/**
 * The operand lengths a gemv of a [rows] by [cols] matrix implies, checked against the [x] and [y] given.
 *
 * Which extent each vector takes is a consequence of [transpose], and deriving it is the same three lines
 * wherever a gemv is entered, so every layer that checks its arguments asks here instead.
 */
internal fun requireGemvShape(rows: Int, cols: Int, transpose: Boolean, x: Int, y: Int) {
    val inputs = if (transpose) rows else cols
    val outputs = if (transpose) cols else rows
    requireShape(x == inputs) { "gemv: x length $x != $inputs" }
    requireShape(y == outputs) { "gemv: y length $y != $outputs" }
}

/** The same check for a caller holding the operand rather than its extents. */
internal fun requireGemvShape(a: Matrix, transpose: Boolean, x: Int, y: Int): Unit =
    requireGemvShape(a.rows, a.cols, transpose, x, y)

/**
 * Both invariants of `op(A)·op(B)` into [c]: that the operands meet, and that the destination is the shape
 * their product has. Nothing is returned, because once this passes the destination's own extents are the
 * product's and the depth is one expression at the call site, which is allocation-free at every call rather
 * than wherever the virtual machine manages to take a record of the three apart again.
 *
 * A caller multiplying the second operand by the first from the right passes them the other way round, since
 * that product is this one with the operands swapped.
 */
internal fun requireGemmShape(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean, c: Matrix): Unit =
    requireGemmShape(a.rows, a.cols, transposeA, b, transposeB, c)

/**
 * The shapes a fresh sparse product needs, which is that the two oriented operands meet. There is no
 * destination to check against: a product with a result of its own takes the extents that follow.
 */
internal fun requireSparseProductShape(a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean) {
    val aRows = if (transposeA) a.cols else a.rows
    val aCols = if (transposeA) a.rows else a.cols
    val bRows = if (transposeB) b.cols else b.rows
    val bCols = if (transposeB) b.rows else b.cols
    requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
}

/**
 * The same check for a caller holding the first operand's extents rather than the operand, which the sparse
 * bindings do: what they hold is a descriptor of a matrix the library owns.
 */
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

/** Rejects a shape no storage can hold, before an allocation turns it into an arithmetic error. */
internal fun requireNonNegativeShape(rows: Int, cols: Int) {
    requireShape(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
}

/** Bounds for a vector position. The message builds only on the failing path, since [requireIndex] inlines. */
internal fun requireInBounds(i: Int, size: Int) {
    requireIndex(i in 0 until size) { "index $i outside [0,$size)" }
}

/** Bounds for a matrix position, both indices at once so one failure names the pair. */
internal fun requireInBounds(i: Int, j: Int, rows: Int, cols: Int) {
    requireIndex(i in 0 until rows && j in 0 until cols) { "index ($i;$j) outside ${rows}x$cols" }
}
