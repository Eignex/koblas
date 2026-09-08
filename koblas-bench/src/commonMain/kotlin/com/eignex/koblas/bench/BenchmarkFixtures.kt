package com.eignex.koblas.bench

import com.eignex.koblas.*
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

internal fun randomMatrix(rows: Int, cols: Int, rng: Random): DenseMatrix =
    DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

internal fun dominantMatrix(n: Int, rng: Random): DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

internal fun lowerSymmetricMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix.zero(n, n)
    for (i in 0 until n) for (j in 0..i) a[i, j] = rng.nextDouble(-1.0, 1.0)
    return a
}

internal fun indefiniteMatrix(n: Int, rng: Random): DenseMatrix =
    symmetricMatrix(n, rng) { i -> if (i % 2 == 0) n.toDouble() else -n.toDouble() }

internal fun spdMatrix(n: Int, rng: Random): DenseMatrix = symmetricMatrix(n, rng) { n.toDouble() }

private inline fun symmetricMatrix(n: Int, rng: Random, shift: (Int) -> Double): DenseMatrix {
    val a = DenseMatrix.zero(n, n)
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

internal fun sparseDominantMatrix(n: Int, rng: Random): SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to (rng.nextDouble(-1.0, 1.0) + n))
        for (i in 0 until n) {
            if (i != j && rng.nextDouble() < SPARSE_DENSITY) entries.add(i to rng.nextDouble(-1.0, 1.0))
        }
        entries
    }
    return SparseMatrix.ofColumns(n, n, columns)
}

/** Sparse parity shapes: regular random, fixed band, and a few unusually long columns. */
internal fun sparseComparisonMatrix(n: Int, density: Double, shape: String, rng: Random): SparseMatrix {
    return sparseComparisonMatrix(n, n, density, shape, rng)
}

internal fun sparseComparisonMatrix(rows: Int, cols: Int, density: Double, shape: String, rng: Random): SparseMatrix {
    val columns = List(cols) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        if (j < rows) entries.add(j to (maxOf(rows, cols) + 1.0))
        when (shape) {
            "regular" -> for (i in 0 until rows) {
                if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-1.0, 1.0))
            }
            "banded" -> for (i in maxOf(0, j - 3)..minOf(rows - 1, j + 3)) {
                if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
            }
            "skewed" -> if (j % maxOf(1, cols / 16) == 0) {
                for (i in 0 until rows) if (i != j && rng.nextDouble() < maxOf(density, 0.25)) {
                    entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            } else {
                repeat(maxOf(1, (rows * density / 4).toInt())) {
                    val i = rng.nextInt(rows)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }
            else -> error("unknown sparse benchmark shape: $shape")
        }
        entries
    }
    return SparseMatrix.ofColumns(rows, cols, columns)
}

/**
 * A sparse symmetric positive-definite matrix as the lower triangle a Cholesky reads, diagonally dominant so
 * it factors without an ordering. The off-diagonal pattern is symmetric by construction, which is what makes
 * the stored triangle describe a symmetric matrix at all.
 */
internal fun sparseSpdMatrix(n: Int, rng: Random): SparseMatrix {
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
    return SparseMatrix.ofColumns(
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

/**
 * A banded symmetric positive-definite matrix, as the lower triangle a Cholesky reads.
 *
 * The band keeps the entries per column fixed as `n` grows, where [sparseSpdMatrix] keeps their density
 * fixed and so grows them. That is what separates a factorization the numeric sweep dominates from one the
 * symbolic pass does, and a symbolic analysis held across refactorizations is worth having only in the
 * second.
 */
internal fun sparseBandedSpdMatrix(n: Int, bandwidth: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
    n,
    n,
    List(n) { j ->
        val below = (j + 1..minOf(j + bandwidth, n - 1)).map { it to rng.nextDouble(-1.0, 1.0) }
        // Dominant over both triangles: a column carries its own band below and its mirror above.
        listOf(j to 2.0 * bandwidth + 1.0) + below
    },
)

/** A tall sparse matrix with full column rank, the shape a least-squares QR is for. */
internal fun sparseTallMatrix(rows: Int, cols: Int, rng: Random): SparseMatrix {
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
    return SparseMatrix.ofColumns(rows, cols, columns)
}

internal fun randomSparseVector(n: Int, density: Double, rng: Random): SparseVector {
    val indices = ArrayList<Int>()
    val values = ArrayList<Double>()
    for (i in 0 until n) {
        if (rng.nextDouble() < density) {
            indices.add(i)
            values.add(rng.nextDouble(-1.0, 1.0))
        }
    }
    return SparseVector.wrap(n, indices.toIntArray(), values.toDoubleArray())
}

internal fun simplexBasis(
    n: Int,
    rng: Random,
    slackFraction: Double = 0.55,
    spikeFraction: Double = 0.08,
    columnNonzeros: Int = 6,
): SparseMatrix {
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
    return SparseMatrix.ofColumns(n, n, columns)
}

/**
 * `[B | I]`, the shape a basis solver draws from: the structural columns of [simplexBasis] and then the
 * logical ones, so a basis is a choice among `2n` columns and a pivot swaps one for another.
 */
internal fun simplexProblem(n: Int, rng: Random, spikeFraction: Double = 0.08): SparseMatrix {
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
    return SparseMatrix.ofColumns(n, 2 * n, columns)
}

internal fun bandUpperTriangle(n: Int): SparseMatrix {
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
    return SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, values)
}
