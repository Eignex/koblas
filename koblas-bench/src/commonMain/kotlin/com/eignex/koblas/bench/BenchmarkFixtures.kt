package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.installLinearAlgebra
import com.eignex.koblas.koblasInfo
import kotlin.random.Random

/**
 * Shared setup for every benchmark in this module: backend selection and the seeded operands.
 *
 * Each benchmark class declares a `backend` parameter and calls [installBackend] from its `@Setup`, so
 * one run compares the native backend against the portable kernels on identical inputs. The operand
 * builders take an explicit [Random] so a class that needs several distinct matrices threads one
 * generator through them all and stays reproducible.
 */

/** Every operand derives from this, so a re-run measures the same numbers rather than similar ones. */
internal const val BENCH_SEED = 20260730

/** The `backend` parameter value that pins a run to the portable kernels. */
internal const val REFERENCE_BACKEND = "reference"

/**
 * The platform's native backend, or `null` to leave resolution to discovery.
 *
 * The JVM returns `null` because its ServiceLoader discovery already installs OpenBLAS from the runtime
 * classpath. Native returns the CBLAS backend explicitly instead of trusting its eager-initialization
 * hook: that hook is an unreferenced top-level property, which the linker is free to drop from a
 * benchmark binary, and a silently unregistered backend would make this compare the reference against
 * itself.
 */
internal expect fun nativeBackend(): LinearAlgebra?

/**
 * Turns the platform's host level-1 kernels on or off, reporting whether any are now installed.
 *
 * The level-1 primitives sit below the [LinearAlgebra] seam, so the `backend` parameter does not reach
 * them and they need their own switch. The JVM has no such seam and always reports `false`.
 */
internal expect fun useHostLevel1(enabled: Boolean): Boolean

/**
 * Installs the backend named by a `backend` parameter and prints what resolved.
 *
 * [REFERENCE_BACKEND] forces the portable kernels; anything else (`auto`) takes the platform's native
 * backend if it has one. The print is not decoration: a missing native library or a withheld
 * `jdk.incubator.vector` silently changes what is being measured, and the resolved line is the only
 * place that shows up.
 */
internal fun installBackend(backend: String) {
    installLinearAlgebra(if (backend == REFERENCE_BACKEND) ReferenceLinearAlgebra else nativeBackend())
    println("resolved: $koblasInfo")
}

/** A general matrix with entries in `[-1, 1)`. */
internal fun randomMatrix(rows: Int, cols: Int, rng: Random): DenseMatrix =
    DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

/**
 * A diagonally dominant `n x n` matrix, well-conditioned enough that factor-and-solve timings are not
 * distorted by pivot searches hunting through near-ties.
 */
internal fun dominantMatrix(n: Int, rng: Random): DenseMatrix {
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    return a
}

/**
 * A matrix whose lower triangle is populated and whose upper triangle is left at zero.
 *
 * `symv` and `symm` read one triangle and infer the other, so an unpopulated upper triangle is what a
 * caller of those routines actually passes, and it catches a backend that reads the wrong side.
 */
internal fun lowerSymmetricMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix(n, n)
    for (i in 0 until n) for (j in 0..i) a[i, j] = rng.nextDouble(-1.0, 1.0)
    return a
}

/**
 * A fully populated symmetric matrix with mixed-sign diagonal shifts, so the LDL factorization has to
 * take the 2x2 Bunch-Kaufman pivots rather than running the diagonal path throughout.
 */
internal fun indefiniteMatrix(n: Int, rng: Random): DenseMatrix {
    val a = DenseMatrix(n, n)
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

/** A vector with entries in `[-1, 1)`. */
internal fun randomVector(n: Int, rng: Random): DoubleArray = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

/** A sparse `n x n` matrix with a dominant diagonal and [density] fill off it. */
internal fun sparseDominantMatrix(n: Int, density: Double, rng: Random): SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to (rng.nextDouble(-1.0, 1.0) + n))
        for (i in 0 until n) if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-1.0, 1.0))
        entries
    }
    return SparseMatrix.ofColumns(n, n, columns)
}
