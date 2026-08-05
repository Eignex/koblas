package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.RoutedVectorKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertTrue

// Comparisons and operand builders shared by the suites in this source set.
//
// The builders that return a pair produce a poisoned input alongside an explicit twin. The poisoned copy
// fills every entry a routine is not allowed to read with NaN, and the twin holds what those entries
// logically are (a mirrored triangle, a zero, a unit diagonal) so a dense routine can compute the
// expected result. NaN propagates through arithmetic, so a routine that strays outside its documented
// triangle fails the value comparison rather than passing quietly.

/** Tolerance for a chain of a few dozen flops, relative to the larger of the expected value and one. */
internal const val TIGHT_TOLERANCE = 1e-12

/** Asserts that [actual] is within [tolerance] of [expected], relative to `max(1, |expected|)`. */
internal fun assertClose(expected: Double, actual: Double, context: String, tolerance: Double = TIGHT_TOLERANCE) {
    assertTrue(
        abs(expected - actual) <= tolerance * maxOf(1.0, abs(expected)),
        "$context: expected $expected actual $actual",
    )
}

/** Asserts [assertClose] entry by entry, reporting the first index that disagrees. */
internal fun assertClose(
    expected: DoubleArray,
    actual: DoubleArray,
    context: String,
    tolerance: Double = TIGHT_TOLERANCE,
) {
    assertTrue(expected.size == actual.size, "$context: size ${actual.size}, expected ${expected.size}")
    for (i in expected.indices) assertClose(expected[i], actual[i], "$context index $i", tolerance)
}

/** Asserts [assertClose] entry by entry over two matrices of the same shape. */
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

/** A vector with entries in `[-1, 1)`. */
internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

/** A general matrix with entries in `[-1, 1)`. */
internal fun randomMatrix(rows: Int, cols: Int, rng: Random): DenseMatrix =
    DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

/**
 * A diagonally dominant matrix, which is nonsingular and well enough conditioned that a solve residual
 * reflects the routine rather than the operand.
 */
internal fun wellConditioned(n: Int, rng: Random): DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/**
 * A symmetric matrix as `(full, poisoned)`: the full copy holds both triangles, the poisoned copy holds
 * only the triangle selected by [lower] and NaN off it. Routines like `symv` and `symm` take the
 * poisoned copy and must agree with the general routine applied to the full one.
 */
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

/**
 * A triangular matrix as `(poisoned, explicit)`. The poisoned copy holds the strict triangle selected by
 * [lower], NaN outside it, and NaN on the diagonal when [unitDiag] says the diagonal is implicitly one.
 * The explicit copy is the same matrix written out in full, so a general routine reproduces the result.
 */
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
                    // Away from zero, so triangular solves stay well conditioned.
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
 * Runs [block] against an empty registry, then puts back whatever was resolved before.
 *
 * The registry is global process state, and on the targets that ship a host backend it is already populated
 * before any test runs — the native ones register eagerly, before `main`, so `registerPlatformBackends` is a
 * no-op there and discovery cannot be asked to run again. A test that clears the registry and does not
 * restore it therefore destroys the platform's backend for every test that follows, which is exactly the
 * failure this exists to prevent: it showed up as a host-BLAS conformance test finding `reference` installed.
 *
 * Restoring goes through [registerBackend] rather than an install, so the result is a genuinely resolved
 * registry rather than a frozen override.
 */
internal fun withCleanBackends(block: () -> Unit) {
    val before = koblas
    val incumbents = buildList {
        add(before.blas)
        add(before.lapack)
        add(before.sparseBlas)
        add(before.sparseLapack)
        add(before.sparseVectorKernels)
        // The context's vector kernels are the routed pair, not the registered object, so the host half has
        // to be unwrapped to be re-offered.
        (before.vectorKernels as? RoutedVectorKernels)?.host?.let { add(it) }
    }.distinct().filter { it !== ReferenceLinearAlgebra && it !== ReferenceSparseLinearAlgebra }
    resetBackends()
    try {
        block()
    } finally {
        resetBackends()
        incumbents.forEach { registerBackend(it) }
    }
}
