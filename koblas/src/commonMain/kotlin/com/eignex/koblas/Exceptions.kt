package com.eignex.koblas

import com.eignex.koblas.*
import kotlin.jvm.JvmInline

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
internal fun requireSquare(a: MatrixLike, what: String) {
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
internal fun requireSolveShapes(rows: Int, cols: Int, b: DenseMatrix, out: DenseMatrix) {
    requireShape(b.rows == rows) { "solve: B has ${b.rows} rows, expected $rows" }
    requireShape(out.rows == cols && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${cols}x${b.cols}"
    }
}

/**
 * The lengths a gemv's operands must have, for a caller that needs them after the check.
 *
 * Both extents ride in one `Long` so the check hands them back without a heap object. A record here would
 * be allocated on every gemv, which the JVM removes only once the caller has reached the top compilation
 * tier and Kotlin/Native never removes.
 */
@JvmInline
internal value class GemvShape(private val packed: Long) {
    constructor(inputs: Int, outputs: Int) : this(packExtents(inputs, outputs))

    val inputs: Int get() = firstExtent(packed)
    val outputs: Int get() = secondExtent(packed)
}

private fun packExtents(first: Int, second: Int): Long =
    (first.toLong() shl Int.SIZE_BITS) or (second.toLong() and INT_MASK)

private fun firstExtent(packed: Long): Int = (packed ushr Int.SIZE_BITS).toInt()

private fun secondExtent(packed: Long): Int = packed.toInt()

private const val INT_MASK = 0xFFFF_FFFFL

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
internal fun requireGemvShape(a: MatrixLike, transpose: Boolean, x: Int, y: Int): GemvShape =
    requireGemvShape(a.rows, a.cols, transpose, x, y)

/**
 * The extents a gemm derives from its operands: `op(A)` is [m] by [k] and `op(B)` is [k] by [n].
 *
 * Three extents do not fit one `Long` the way [GemvShape] packs two, and a gemm's work dwarfs one record.
 */
internal data class GemmShape(val m: Int, val k: Int, val n: Int)

/**
 * The extents `op(A)·op(B)` into [c] implies, with both invariants checked: that the operands meet, and that
 * the destination is the shape their product has.
 *
 * A caller multiplying the second operand by the first from the right passes them the other way round, since
 * that product is this one with the operands swapped.
 */
internal fun requireGemmShape(
    a: MatrixLike,
    transposeA: Boolean,
    b: MatrixLike,
    transposeB: Boolean,
    c: MatrixLike,
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
    b: MatrixLike,
    transposeB: Boolean,
    c: MatrixLike,
): GemmShape {
    val m = if (transposeA) aCols else aRows
    val k = if (transposeA) aRows else aCols
    val kB = if (transposeB) b.cols else b.rows
    val n = if (transposeB) b.rows else b.cols
    requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
    requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
    return GemmShape(m, k, n)
}

/**
 * The order and depth a `syrk` or `syr2k` works over, after checking C against them.
 *
 * Packed into one `Long` for the same reason as [GemvShape]: a syrk sits on per-observation update paths.
 */
@JvmInline
internal value class SyrkShape(private val packed: Long) {
    constructor(order: Int, depth: Int) : this(packExtents(order, depth))

    val order: Int get() = firstExtent(packed)
    val depth: Int get() = secondExtent(packed)

    operator fun component1(): Int = order
    operator fun component2(): Int = depth
}

/** [SyrkShape] for [a] under [transpose], having checked that C is square and matches the order. */
internal fun requireSyrkShape(a: DenseMatrix, transpose: Boolean, c: DenseMatrix, what: String): SyrkShape {
    val n = if (transpose) a.cols else a.rows
    val k = if (transpose) a.rows else a.cols
    requireShape(c.rows == n && c.cols == n) { "$what: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
    return SyrkShape(n, k)
}

/** [SyrkShape] for a `syr2k`, having checked B against A and C against the order. */
internal fun requireSyr2kShape(
    a: DenseMatrix,
    b: DenseMatrix,
    transpose: Boolean,
    c: DenseMatrix,
    what: String,
): SyrkShape {
    requireShape(b.rows == a.rows && b.cols == a.cols) {
        "$what: B is ${b.rows}x${b.cols}, expected ${a.rows}x${a.cols} to match A"
    }
    return requireSyrkShape(a, transpose, c, what)
}

/** Checks the symmetric matrix of a `syr` against its vector, returning its dimension. */
internal fun requireSyrShape(a: MatrixLike, x: Int, what: String): Int {
    requireSquare(a, what)
    val n = a.rows
    requireShape(x == n) { "$what: x length $x != $n" }
    return n
}

/** Checks the symmetric matrix of a `syr2` against both vectors, returning its dimension. */
internal fun requireSyr2Shape(a: MatrixLike, x: Int, y: Int, what: String): Int {
    requireSquare(a, what)
    val n = a.rows
    requireShape(x == n && y == n) { "$what: operand lengths $x and $y must both be $n" }
    return n
}

/** Checks a symmetric matrix against the two vectors of a `symv`, returning its dimension. */
internal fun requireSymvShape(a: DenseMatrix, x: Int, y: Int): Int {
    requireSquare(a, "symv")
    val n = a.rows
    requireShape(x == n) { "symv: x length $x != $n" }
    requireShape(y == n) { "symv: y length $y != $n" }
    return n
}

/** Checks the triangle and the block of a `trsm` or `trmm`, returning the triangle's dimension. */
internal fun requireTriangularMatrixShape(a: MatrixLike, b: DenseMatrix, right: Boolean, what: String): Int {
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
