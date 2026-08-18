package com.eignex.koblas.bench

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.F64SparseVector
import kotlin.random.Random

/** Every operand derives from this, so a re-run measures the same numbers rather than similar ones. */
internal const val BENCH_SEED = 20260730

/** Off-diagonal fill fraction for the sparse operands, low enough that the sparse paths stay sparse. */
internal const val SPARSE_DENSITY = 0.01

/** A scale just off one, so repeated in-place updates neither fold away nor drift out of range. */
internal const val NEAR_UNIT_SCALE = 1.000001

/** The shape parameter value that takes a simplex-like basis. */
internal const val BASIS_SHAPE = "basis"

/** The shape parameter value that takes a matrix with uniformly scattered fill. */
internal const val RANDOM_SHAPE = "random"

/** The generator every fixture draws from, seeded so all suites see the same operands. */
internal fun benchRng(): Random = Random(BENCH_SEED)

/** A general matrix with entries uniform in -1 to 1. */
internal fun randomMatrix(rows: Int, cols: Int, rng: Random): F64DenseMatrix =
    F64DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

/** Diagonally dominant, so pivot searches through near-ties do not distort factor-and-solve timings. */
internal fun dominantMatrix(n: Int, rng: Random): F64DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/** A matrix whose lower triangle is populated and whose upper triangle is left at zero. */
internal fun lowerSymmetricMatrix(n: Int, rng: Random): F64DenseMatrix {
    val a = F64DenseMatrix.zero(n, n)
    for (i in 0 until n) for (j in 0..i) a[i, j] = rng.nextDouble(-1.0, 1.0)
    return a
}

/** Mixed-sign diagonal shifts force LDL onto the 2x2 Bunch-Kaufman pivots rather than the diagonal path. */
internal fun indefiniteMatrix(n: Int, rng: Random): F64DenseMatrix =
    symmetricMatrix(n, rng) { i -> if (i % 2 == 0) n.toDouble() else -n.toDouble() }

/** Symmetric positive-definite, built by symmetrizing and then dominating the diagonal, not as A·Aᵀ. */
internal fun spdMatrix(n: Int, rng: Random): F64DenseMatrix = symmetricMatrix(n, rng) { n.toDouble() }

/** A full symmetric matrix with entries uniform in -1 to 1, plus [shift] added to each diagonal entry. */
private inline fun symmetricMatrix(n: Int, rng: Random, shift: (Int) -> Double): F64DenseMatrix {
    val a = F64DenseMatrix.zero(n, n)
    for (i in 0 until n) {
        for (j in 0..i) {
            val v = rng.nextDouble(-1.0, 1.0)
            a[i, j] = v
            a[j, i] = v
        }
        a[i, i] = a[i, i] + shift(i)
    }
    return a
}

/** A vector with entries uniform in -1 to 1. */
internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

/** A dominant diagonal with [SPARSE_DENSITY] fill off it. */
internal fun sparseDominantMatrix(n: Int, rng: Random): F64SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to (rng.nextDouble(-1.0, 1.0) + n))
        for (i in 0 until n) {
            if (i != j && rng.nextDouble() < SPARSE_DENSITY) entries.add(i to rng.nextDouble(-1.0, 1.0))
        }
        entries
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/** Symmetric with a dominant diagonal, so the sparse LDL and Cholesky paths have a matrix they accept. */
internal fun sparseSpdMatrix(n: Int, rng: Random): F64SparseMatrix {
    val entries = Array(n) { HashMap<Int, Double>() }
    for (i in 0 until n) entries[i][i] = rng.nextDouble(-1.0, 1.0) + n
    for (j in 0 until n) {
        for (i in 0 until j) {
            if (rng.nextDouble() < SPARSE_DENSITY) {
                val v = rng.nextDouble(-1.0, 1.0)
                entries[j][i] = v
                entries[i][j] = v
            }
        }
    }
    return F64SparseMatrix.ofColumns(n, n, List(n) { j -> entries[j].map { (i, v) -> i to v } })
}

/** A sparse vector with about `density * n` stored entries, at ascending positions. */
internal fun randomSparseVector(n: Int, density: Double, rng: Random): F64SparseVector {
    val indices = ArrayList<Int>()
    val values = ArrayList<Double>()
    for (i in 0 until n) {
        if (rng.nextDouble() < density) {
            indices.add(i)
            values.add(rng.nextDouble(-1.0, 1.0))
        }
    }
    return F64SparseVector.wrap(n, indices.toIntArray(), values.toDoubleArray())
}

/**
 * A structural stand-in for a simplex basis: near-triangular, mostly slack columns, a few spikes.
 *
 * @param n the basis dimension.
 * @param slackFraction how many columns are unit vectors, as a fraction of [n].
 * @param spikeFraction how many columns violate the triangular order, as a fraction of [n].
 * @param columnNonzeros entries in a structural column, before the triangular restriction.
 */
internal fun simplexBasis(
    n: Int,
    rng: Random,
    slackFraction: Double = 0.55,
    spikeFraction: Double = 0.08,
    columnNonzeros: Int = 6,
): F64SparseMatrix {
    val slacks = (n * slackFraction).toInt()
    val spikes = (n * spikeFraction).toInt()
    val isSpike = BooleanArray(n)
    repeat(spikes) { isSpike[rng.nextInt(n)] = true }
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to (1.0 + rng.nextDouble()))
        when {
            j < slacks -> Unit // a slack column is the unit vector, and stays one
            isSpike[j] -> {
                repeat(columnNonzeros) {
                    val i = rng.nextInt(n)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }

            else -> {
                repeat(columnNonzeros) {
                    val i = rng.nextInt(j + 1)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }
        }
        entries
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/** The upper triangle of a tridiagonal band of dimension [n]. */
internal fun bandUpperTriangle(n: Int): F64SparseMatrix {
    val rowIdx = IntArray(2 * n - 1)
    val colIdx = IntArray(2 * n - 1)
    val values = DoubleArray(2 * n - 1)
    var k = 0
    for (j in 0 until n) {
        if (j > 0) {
            rowIdx[k] = j - 1
            colIdx[k] = j
            values[k] = -1.0
            k++
        }
        rowIdx[k] = j
        colIdx[k] = j
        values[k] = 4.0
        k++
    }
    return F64SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, values)
}

/** The upper triangle of a 5-point Laplacian on the squarest grid with at most [n] points. */
internal fun gridUpperTriangle(n: Int): F64SparseMatrix {
    var side = 1
    while ((side + 1) * (side + 1) <= n) side++
    val points = side * side
    val rowIdx = ArrayList<Int>()
    val colIdx = ArrayList<Int>()
    val values = ArrayList<Double>()
    for (y in 0 until side) {
        for (x in 0 until side) {
            val i = y * side + x
            rowIdx.add(i)
            colIdx.add(i)
            values.add(8.0)
            if (x + 1 < side) {
                rowIdx.add(i)
                colIdx.add(i + 1)
                values.add(-1.0)
            }
            if (y + 1 < side) {
                rowIdx.add(i)
                colIdx.add(i + side)
                values.add(-1.0)
            }
        }
    }
    return F64SparseMatrix.ofTriplets(points, points, rowIdx.toIntArray(), colIdx.toIntArray(), values.toDoubleArray())
}
