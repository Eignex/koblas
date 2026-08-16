package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import kotlin.random.Random

/** Every operand derives from this, so a re-run measures the same numbers rather than similar ones. */
internal const val BENCH_SEED = 20260730

/** The backend parameter value that pins a run to the portable kernels. */
internal const val REFERENCE_BACKEND = "reference"

/** The platform's native backend, or null to leave resolution to discovery. */
internal expect fun nativeBackend(): LinearAlgebra?

/**
 * Turns the platform's host level-1 kernels on or off, reporting whether any are now installed. They sit
 * below the [LinearAlgebra] seam, so the backend parameter does not reach them.
 */
internal expect fun useHostLevel1(enabled: Boolean): Boolean

/**
 * Installs the named backend and prints what resolved. [REFERENCE_BACKEND] forces the portable kernels,
 * anything else takes the platform's native backend if it has one.
 */
internal fun installBackend(backend: String) {
    val chosen = if (backend == REFERENCE_BACKEND) ReferenceLinearAlgebra else nativeBackend()
    installBackends(chosen?.let { koblas.with(blas = it, lapack = it) })
    println("resolved: $koblasInfo")
}

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
internal fun indefiniteMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix.zero(n, n)
    for (i in 0 until n) {
        for (j in 0..i) {
            var v = rng.nextDouble(-1.0, 1.0)
            if (i == j) v += if (i % 2 == 0) n.toDouble() else -n.toDouble()
            a[i, j] = v
            a[j, i] = v
        }
    }
    return a
}

/** Symmetric positive-definite, built by symmetrizing and then dominating the diagonal, not as A·Aᵀ. */
internal fun spdMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix.zero(n, n)
    for (i in 0 until n) {
        for (j in 0..i) {
            val v = rng.nextDouble(-1.0, 1.0)
            a[i, j] = v
            a[j, i] = v
        }
        a[i, i] = a[i, i] + n
    }
    return a
}

/** A vector with entries uniform in -1 to 1. */
internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

/** A dominant diagonal with [density] fill off it. */
internal fun sparseDominantMatrix(n: Int, density: Double, rng: Random): SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to (rng.nextDouble(-1.0, 1.0) + n))
        for (i in 0 until n) if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-1.0, 1.0))
        entries
    }
    return SparseMatrix.ofColumns(n, n, columns)
}

/** Symmetric with a dominant diagonal, so the sparse LDL and Cholesky paths have a matrix they accept. */
internal fun sparseSpdMatrix(n: Int, density: Double, rng: Random): SparseMatrix {
    val entries = Array(n) { HashMap<Int, Double>() }
    for (i in 0 until n) entries[i][i] = rng.nextDouble(-1.0, 1.0) + n
    for (j in 0 until n) {
        for (i in 0 until j) {
            if (rng.nextDouble() < density) {
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
