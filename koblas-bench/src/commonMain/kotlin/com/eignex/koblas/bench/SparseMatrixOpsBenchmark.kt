package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseMatrixOpsBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: SparseMatrix
    private lateinit var factors: DoubleArray

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        factors = DoubleArray(n) { NEAR_UNIT_SCALE }
    }

    @Benchmark
    fun norm1(): Double = a.norm1()

    @Benchmark
    fun normInf(): Double = a.normInf()

    @Benchmark
    fun normFro(): Double = a.normFro()

    @Benchmark
    fun scaleRowsBench(): SparseMatrix {
        a.scaleRows(factors)
        return a
    }

    @Benchmark
    fun scaleColumnsBench(): SparseMatrix {
        a.scaleColumns(factors)
        return a
    }

    @Benchmark
    fun column(): SparseVector = a.column(n / 2)

    @Benchmark
    fun row(): SparseVector = a.row(n / 2)
}
