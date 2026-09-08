package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

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

    @Benchmark
    fun norm1(): Double = a.norm1()

    @Benchmark
    fun normInf(): Double = a.normInf()

    @Benchmark
    fun normFro(): Double = a.normFro()

    @Benchmark
    fun scaleRowsBench(): DenseMatrix {
        a.scaleRows(factors)
        return a
    }

    @Benchmark
    fun scaleColumnsBench(): DenseMatrix {
        a.scaleColumns(factors)
        return a
    }

    @Benchmark
    fun transposeDense(): DenseMatrix = a.transpose()
}
