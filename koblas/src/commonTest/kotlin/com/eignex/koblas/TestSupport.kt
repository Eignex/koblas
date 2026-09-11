package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/** Tolerance for a chain of a few dozen flops, relative to `max(1, |expected|)`. */
internal const val TIGHT_TOLERANCE = 1e-12

/** A non-koblas matrix storage used to exercise public adapter fallbacks. */
internal class ForeignSpdMatrix(override val rows: Int) : Matrix {
    override val cols: Int get() = rows
    override fun get(i: Int, j: Int): Double = if (i == j) rows + 2.0 else 1.0 / (1 + i + j)
    override fun toArray(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> this[i, j] } }
}

/** A non-koblas vector storage used to exercise public adapter fallbacks. */
internal class ForeignRampVector(override val size: Int) : Vector {
    override fun get(i: Int): Double = i * 0.5 - 1.0
    override fun toDoubleArray(): DoubleArray = DoubleArray(size) { this[it] }
}

internal fun assertClose(expected: Double, actual: Double, context: String, tolerance: Double = TIGHT_TOLERANCE) {
    if (expected.isNaN()) fail("$context: the numerical oracle produced NaN; assert that contract explicitly")
    if (actual.isNaN()) fail("$context: expected $expected actual NaN")
    if (expected == actual) return
    if (!expected.isFinite() || !actual.isFinite()) fail("$context: expected $expected actual $actual")
    val error = abs(expected - actual)
    val bound = tolerance * maxOf(1.0, abs(expected))
    assertTrue(
        error <= bound,
        "$context: expected $expected actual $actual error $error tolerance $bound",
    )
}

internal fun assertClose(
    expected: DoubleArray,
    actual: DoubleArray,
    context: String,
    tolerance: Double = TIGHT_TOLERANCE,
) {
    assertEquals(
        expected.size,
        actual.size,
        "$context: size ${actual.size}, expected ${expected.size}",
    )
    for (i in expected.indices) assertClose(expected[i], actual[i], "$context index $i", tolerance)
}

internal fun assertClose(
    expected: DenseMatrix,
    actual: DenseMatrix,
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

internal fun randomMatrix(rows: Int, cols: Int, rng: Random): DenseMatrix =
    DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

/** CSC fixture with empty columns, irregular rows, and an explicitly stored zero. */
internal fun sparseStorageExample(): SparseMatrix = SparseMatrix.ofTriplets(
    rows = 3,
    cols = 4,
    rowIdx = intArrayOf(0, 2, 1, 0, 2),
    colIdx = intArrayOf(0, 0, 1, 3, 3),
    values = doubleArrayOf(1.0, -2.0, 3.0, 0.0, -4.0),
)

internal fun SparseMatrix.denseCopy(): DenseMatrix = DenseMatrix.ofRows(toArray())

internal fun wellConditioned(n: Int, rng: Random): DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/** `(full, poisoned)`, where the poisoned copy holds only the triangle selected by [lower], NaN off it. */
internal fun poisonedSymmetric(rng: Random, n: Int, lower: Boolean): Pair<DenseMatrix, DenseMatrix> {
    val full = DenseMatrix(n)
    val poisoned = DenseMatrix(n)
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

/** `(poisoned, explicit)`, poisoned outside the [lower] triangle and on an implicit unit diagonal. */
internal fun poisonedTriangle(rng: Random, n: Int, lower: Boolean, unitDiag: Boolean): Pair<DenseMatrix, DenseMatrix> {
    val poisoned = DenseMatrix(n)
    val explicit = DenseMatrix(n)
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
