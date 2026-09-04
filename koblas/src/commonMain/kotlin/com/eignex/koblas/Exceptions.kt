package com.eignex.koblas

import com.eignex.koblas.core.*

/** Operands whose shapes do not fit the routine. */
public class DimensionMismatch(message: String) : IllegalArgumentException(message)

/** A numerical outcome that prevents a requested factorization or solve from completing. */
public sealed class KoblasException(message: String) : ArithmeticException(message)

/**
 * A factorization met an exactly zero pivot, so the matrix it came from has no inverse.
 *
 * @property position the 0-based pivot index where the factorization failed, LAPACK's `info` less one.
 * @param message what failed, naming the routine and the position.
 */
public class SingularMatrix(public val position: Int, message: String) : KoblasException(message)

/**
 * A Cholesky or a strict `L·D·Lᵀ` met a pivot that was zero, negative or NaN.
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
internal fun requireSquare(a: F64MatrixLike, what: String) {
    requireShape(a.rows == a.cols) { "$what requires a square matrix; got ${a.rows}x${a.cols}" }
}

/** The conformance two vector operands need, reported in the order the caller compared them. */
internal fun requireSameSize(a: Int, b: Int) {
    requireShape(a == b) { "size mismatch: $a vs $b" }
}

/**
 * The shapes a solve needs: a right-hand side of [rows] entries and a destination of [cols].
 *
 * The two differ only for a least-squares solve, where `A` is taller than it is wide. A square factorization
 * passes its order twice, which is what makes this one check rather than one per shape of factorization.
 */
internal fun requireSolveShapes(rows: Int, cols: Int, b: DoubleArray, out: DoubleArray) {
    requireShape(b.size == rows) { "solve: b size ${b.size}, expected $rows" }
    requireShape(out.size == cols) { "solve: out size ${out.size}, expected $cols" }
}

/** The same shapes for a solve over a panel of right-hand sides, which keeps its columns. */
internal fun requireSolveShapes(rows: Int, cols: Int, b: F64DenseMatrix, out: F64DenseMatrix) {
    requireShape(b.rows == rows) { "solve: B has ${b.rows} rows, expected $rows" }
    requireShape(out.rows == cols && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${cols}x${b.cols}"
    }
}

/** The lengths a gemv's operands must have, for a caller that needs them after the check. */
internal class GemvShape(val inputs: Int, val outputs: Int)

/**
 * The operand lengths a gemv of a [rows] by [cols] matrix implies, checked against the [x] and [y] given.
 *
 * Which extent each vector takes is a consequence of [transpose], and deriving it is the same three lines
 * wherever a gemv is entered, so every layer that checks its arguments asks here instead.
 */
internal fun requireGemvShape(rows: Int, cols: Int, transpose: Boolean, x: Int, y: Int): GemvShape {
    val inputs = if (transpose) rows else cols
    val outputs = if (transpose) cols else rows
    requireShape(x == inputs) { "gemv: x length $x != $inputs" }
    requireShape(y == outputs) { "gemv: y length $y != $outputs" }
    return GemvShape(inputs, outputs)
}

/** The same check for a caller holding the operand rather than its extents. */
internal fun requireGemvShape(a: F64MatrixLike, transpose: Boolean, x: Int, y: Int): GemvShape =
    requireGemvShape(a.rows, a.cols, transpose, x, y)

/** The extents a gemm derives from its operands: `op(A)` is [m] by [k] and `op(B)` is [k] by [n]. */
internal data class GemmShape(val m: Int, val k: Int, val n: Int)

/**
 * The extents `op(A)·op(B)` into [c] implies, with both invariants checked: that the operands meet, and that
 * the destination is the shape their product has.
 *
 * A caller multiplying the second operand by the first from the right passes them the other way round, since
 * that product is this one with the operands swapped.
 */
internal fun requireGemmShape(
    a: F64MatrixLike,
    transposeA: Boolean,
    b: F64MatrixLike,
    transposeB: Boolean,
    c: F64MatrixLike,
): GemmShape = requireGemmShape(a.rows, a.cols, transposeA, b, transposeB, c)

/**
 * The same check for a caller holding the first operand's extents rather than the operand, which the sparse
 * bindings do: what they hold is a descriptor of a matrix the library owns.
 */
@Suppress("LongParameterList") // the first operand's extents in place of the operand itself
internal fun requireGemmShape(
    aRows: Int,
    aCols: Int,
    transposeA: Boolean,
    b: F64MatrixLike,
    transposeB: Boolean,
    c: F64MatrixLike,
): GemmShape {
    val m = if (transposeA) aCols else aRows
    val k = if (transposeA) aRows else aCols
    val kB = if (transposeB) b.cols else b.rows
    val n = if (transposeB) b.rows else b.cols
    requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
    requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
    return GemmShape(m, k, n)
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
