package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.norm1
import com.eignex.koblas.normFro
import com.eignex.koblas.normInf
import com.eignex.koblas.scaleColumns
import com.eignex.koblas.scaleRows
import com.eignex.koblas.transpose
import kotlin.random.Random
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/**
 * The matrix-wide operations that are not BLAS calls: the three norms a condition estimate needs, the row
 * and column scalings equilibration applies, and transposition. All are `O(n²)` and memory-bound, so they
 * show the cost of the traversal order rather than of any arithmetic.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class MatrixOpsBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: DenseMatrix
    private lateinit var sparse: SparseMatrix
    private lateinit var factors: DoubleArray

    @Setup
    fun setup() {
        val rng = Random(BENCH_SEED)
        a = randomMatrix(n, n, rng)
        sparse = sparseDominantMatrix(n, density = 0.01, rng = rng)
        factors = DoubleArray(n) { 1.000001 }
    }

    /** Column sums, which run with the column-major layout. */
    @Benchmark
    fun norm1(): Double = norm1(a)

    /** Row sums, which run across it and pay for the stride. */
    @Benchmark
    fun normInf(): Double = normInf(a)

    @Benchmark
    fun normFro(): Double = normFro(a)

    @Benchmark
    fun scaleRowsBench(): DenseMatrix {
        scaleRows(a, factors)
        return a
    }

    @Benchmark
    fun scaleColumnsBench(): DenseMatrix {
        scaleColumns(a, factors)
        return a
    }

    @Benchmark
    fun transposeDense(): DenseMatrix = a.transpose()

    @Benchmark
    fun transposeSparse(): SparseMatrix = sparse.transpose()
}
