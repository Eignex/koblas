package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.norm1
import com.eignex.koblas.normFro
import com.eignex.koblas.normInf
import com.eignex.koblas.scaleColumns
import com.eignex.koblas.scaleRows
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class MatrixOpsBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: F64DenseMatrix
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
    fun scaleRowsBench(): F64DenseMatrix {
        a.scaleRows(factors)
        return a
    }

    @Benchmark
    fun scaleColumnsBench(): F64DenseMatrix {
        a.scaleColumns(factors)
        return a
    }

    @Benchmark
    fun transposeDense(): F64DenseMatrix = a.transpose()
}
