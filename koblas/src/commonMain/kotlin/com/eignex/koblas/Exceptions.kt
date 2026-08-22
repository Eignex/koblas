package com.eignex.koblas

import com.eignex.koblas.core.*
/** What koblas throws when a routine cannot do what was asked. Every one is an [IllegalArgumentException]. */
public sealed class KoblasException(message: String) : IllegalArgumentException(message)

/** Operands whose shapes do not fit the routine. */
public class DimensionMismatch(message: String) : KoblasException(message)

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
