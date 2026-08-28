package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64PivotedQrDecomposition
import com.eignex.koblas.dense.qrPivoted
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class PivotedQrGateBenchmark {
    @Param("4", "8", "12", "16", "32", "64")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var square: F64DenseMatrix

    private lateinit var tall: F64DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        square = randomMatrix(n, n, rng)
        tall = randomMatrix(2 * n, n, rng)
    }

    @Benchmark
    fun pivotedSquare(): F64PivotedQrDecomposition = square.qrPivoted()

    @Benchmark
    fun pivotedTall(): F64PivotedQrDecomposition = tall.qrPivoted()
}
