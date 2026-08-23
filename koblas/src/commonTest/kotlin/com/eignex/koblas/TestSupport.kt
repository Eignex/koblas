package com.eignex.koblas

import com.eignex.koblas.core.*
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertTrue

/** Tolerance for a chain of a few dozen flops, relative to `max(1, |expected|)`. */
internal const val TIGHT_TOLERANCE = 1e-12

internal fun assertClose(expected: Double, actual: Double, context: String, tolerance: Double = TIGHT_TOLERANCE) {
    assertTrue(
        abs(expected - actual) <= tolerance * maxOf(1.0, abs(expected)),
        "$context: expected $expected actual $actual",
    )
}

internal fun assertClose(
    expected: DoubleArray,
    actual: DoubleArray,
    context: String,
    tolerance: Double = TIGHT_TOLERANCE,
) {
    assertTrue(expected.size == actual.size, "$context: size ${actual.size}, expected ${expected.size}")
    for (i in expected.indices) assertClose(expected[i], actual[i], "$context index $i", tolerance)
}

internal fun assertClose(
    expected: F64DenseMatrix,
    actual: F64DenseMatrix,
    context: String,
    tolerance: Double = TIGHT_TOLERANCE,
) {
    assertTrue(
        expected.rows == actual.rows && expected.cols == actual.cols,
        "$context: shape ${actual.rows}x${actual.cols}, expected ${expected.rows}x${expected.cols}",
    )
    assertClose(expected.data, actual.data, context, tolerance)
}

internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

internal fun randomMatrix(rows: Int, cols: Int, rng: Random): F64DenseMatrix =
    F64DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

internal fun wellConditioned(n: Int, rng: Random): F64DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/** `(full, poisoned)`, where the poisoned copy holds only the triangle selected by [lower], NaN off it. */
internal fun poisonedSymmetric(rng: Random, n: Int, lower: Boolean): Pair<F64DenseMatrix, F64DenseMatrix> {
    val full = F64DenseMatrix(n)
    val poisoned = F64DenseMatrix(n)
    for (i in 0 until n) {
        for (j in 0..i) {
            val v = rng.nextDouble(-1.0, 1.0)
            full[i, j] = v
            full[j, i] = v
            poisoned[if (lower) i else j, if (lower) j else i] = v
            if (j != i) poisoned[if (lower) j else i, if (lower) i else j] = Double.NaN
        }
    }
    return full to poisoned
}

/** A random symmetric indefinite matrix as `(full, poisoned)`, whose mixed-sign pivots force 2x2 blocks. */
internal fun poisonedIndefinite(rng: Random, n: Int): Pair<F64DenseMatrix, F64DenseMatrix> {
    val full = F64DenseMatrix(n)
    val poisoned = F64DenseMatrix(n)
    for (i in 0 until n) {
        for (j in 0..i) {
            var v = rng.nextDouble(-1.0, 1.0)
            if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
            full[i, j] = v
            full[j, i] = v
            poisoned[i, j] = v
            if (j != i) poisoned[j, i] = Double.NaN
        }
    }
    return full to poisoned
}

/** `(poisoned, explicit)`, poisoned outside the [lower] triangle and on an implicit unit diagonal. */
internal fun poisonedTriangle(
    rng: Random,
    n: Int,
    lower: Boolean,
    unitDiag: Boolean,
): Pair<F64DenseMatrix, F64DenseMatrix> {
    val poisoned = F64DenseMatrix(n)
    val explicit = F64DenseMatrix(n)
    for (i in 0 until n) {
        for (j in 0 until n) {
            val strict = if (lower) j < i else j > i
            when {
                strict -> {
                    val v = rng.nextDouble(-1.0, 1.0)
                    poisoned[i, j] = v
                    explicit[i, j] = v
                }

                i == j && unitDiag -> {
                    poisoned[i, j] = Double.NaN
                    explicit[i, j] = 1.0
                }

                i == j -> {
                    val v = rng.nextDouble(2.0, 4.0)
                    poisoned[i, j] = v
                    explicit[i, j] = v
                }

                else -> poisoned[i, j] = Double.NaN
            }
        }
    }
    return poisoned to explicit
}

/**
 * Runs [block] against an empty registry, then puts the process back by replaying platform discovery.
 *
 * Discovery is explicit, so clearing the registry without replaying it leaves later tests on the portable
 * fallbacks.
 */
internal fun withCleanBackends(block: () -> Unit) {
    resetBackends()
    try {
        block()
    } finally {
        resetBackends()
        rediscoverBackends()
    }
}
