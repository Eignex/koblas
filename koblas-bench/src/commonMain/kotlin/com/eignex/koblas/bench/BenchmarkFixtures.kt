package com.eignex.koblas.bench

import com.eignex.koblas.core.*
import kotlin.math.abs
import kotlin.random.Random

// A fixed seed makes repeated runs comparable.
internal const val BENCH_SEED = 20260730

internal const val SPARSE_DENSITY = 0.01

// This avoids folding repeated updates away while keeping values in range.
internal const val NEAR_UNIT_SCALE = 1.000001

internal const val BASIS_SHAPE = "basis"

internal const val RANDOM_SHAPE = "random"

internal fun benchRng(): Random = Random(BENCH_SEED)

internal fun randomMatrix(rows: Int, cols: Int, rng: Random): F64DenseMatrix =
    F64DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

internal fun dominantMatrix(n: Int, rng: Random): F64DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

internal fun lowerSymmetricMatrix(n: Int, rng: Random): F64DenseMatrix {
    val a = F64DenseMatrix.zero(n, n)
    for (i in 0 until n) for (j in 0..i) a[i, j] = rng.nextDouble(-1.0, 1.0)
    return a
}

internal fun indefiniteMatrix(n: Int, rng: Random): F64DenseMatrix =
    symmetricMatrix(n, rng) { i -> if (i % 2 == 0) n.toDouble() else -n.toDouble() }

internal fun spdMatrix(n: Int, rng: Random): F64DenseMatrix = symmetricMatrix(n, rng) { n.toDouble() }

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

internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

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

/**
 * A sparse symmetric positive-definite matrix as the lower triangle a Cholesky reads, diagonally dominant so
 * it factors without an ordering. The off-diagonal pattern is symmetric by construction, which is what makes
 * the stored triangle describe a symmetric matrix at all.
 */
internal fun sparseSpdMatrix(n: Int, rng: Random): F64SparseMatrix {
    val below = List(n) { HashMap<Int, Double>() }
    val weight = DoubleArray(n)
    for (j in 0 until n) {
        for (i in j + 1 until n) {
            if (rng.nextDouble() >= SPARSE_DENSITY) continue
            val v = rng.nextDouble(-1.0, 1.0)
            below[j][i] = v
            weight[i] += abs(v)
            weight[j] += abs(v)
        }
    }
    return F64SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            val column = ArrayList<Pair<Int, Double>>()
            column.add(j to weight[j] + 1.0)
            for (i in j + 1 until n) below[j][i]?.let { column.add(i to it) }
            column
        },
    )
}

/** A tall sparse matrix with full column rank, the shape a least-squares QR is for. */
internal fun sparseTallMatrix(rows: Int, cols: Int, rng: Random): F64SparseMatrix {
    val columns = List(cols) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        for (i in 0 until rows) {
            when {
                i == j -> entries.add(i to (rng.nextDouble(-1.0, 1.0) + cols))
                rng.nextDouble() < SPARSE_DENSITY -> entries.add(i to rng.nextDouble(-1.0, 1.0))
            }
        }
        entries
    }
    return F64SparseMatrix.ofColumns(rows, cols, columns)
}

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
            j < slacks -> Unit
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

/**
 * `[B | I]`, the shape a basis solver draws from: the structural columns of [simplexBasis] and then the
 * logical ones, so a basis is a choice among `2n` columns and a pivot swaps one for another.
 */
internal fun simplexProblem(n: Int, rng: Random, spikeFraction: Double = 0.08): F64SparseMatrix {
    val structural = simplexBasis(n, rng, spikeFraction = spikeFraction)
    val columns = List(2 * n) { j ->
        if (j >= n) {
            listOf((j - n) to 1.0)
        } else {
            val entries = ArrayList<Pair<Int, Double>>()
            structural.forEachInColumn(j) { i, v -> entries.add(i to v) }
            entries
        }
    }
    return F64SparseMatrix.ofColumns(n, 2 * n, columns)
}

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
