package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.norm1
import com.eignex.koblas.normFro
import com.eignex.koblas.normInf
import com.eignex.koblas.scaleColumns
import com.eignex.koblas.scaleRows
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

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
    private lateinit var factors: DoubleArray

    @Setup
    fun setup() {
        val rng = benchRng()
        a = randomMatrix(n, n, rng)
        factors = DoubleArray(n) { NEAR_UNIT_SCALE }
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
}
