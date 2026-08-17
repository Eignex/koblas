package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import kotlin.random.Random

/** A general matrix with entries uniform in -1 to 1. */
internal fun randomMatrix(rows: Int, cols: Int, rng: Random): DenseMatrix =
    DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

/** Diagonally dominant, so pivot searches through near-ties do not distort factor-and-solve timings. */
internal fun dominantMatrix(n: Int, rng: Random): DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/** A matrix whose lower triangle is populated and whose upper triangle is left at zero. */
internal fun lowerSymmetricMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix.zero(n, n)
    for (i in 0 until n) for (j in 0..i) a[i, j] = rng.nextDouble(-1.0, 1.0)
    return a
}

/** Mixed-sign diagonal shifts force LDL onto the 2x2 Bunch-Kaufman pivots rather than the diagonal path. */
internal fun indefiniteMatrix(n: Int, rng: Random): DenseMatrix =
    symmetricMatrix(n, rng) { i -> if (i % 2 == 0) n.toDouble() else -n.toDouble() }

/** Symmetric positive-definite, built by symmetrizing and then dominating the diagonal, not as A·Aᵀ. */
internal fun spdMatrix(n: Int, rng: Random): DenseMatrix = symmetricMatrix(n, rng) { n.toDouble() }

/** A full symmetric matrix with entries uniform in -1 to 1, plus [shift] added to each diagonal entry. */
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

/** A vector with entries uniform in -1 to 1. */
internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

/** A dominant diagonal with [SPARSE_DENSITY] fill off it. */
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

/** Symmetric with a dominant diagonal, so the sparse LDL and Cholesky paths have a matrix they accept. */
internal fun sparseSpdMatrix(n: Int, rng: Random): SparseMatrix {
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
    return SparseMatrix.ofColumns(n, n, List(n) { j -> entries[j].map { (i, v) -> i to v } })
}

/** A sparse vector with about `density * n` stored entries, at ascending positions. */
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
