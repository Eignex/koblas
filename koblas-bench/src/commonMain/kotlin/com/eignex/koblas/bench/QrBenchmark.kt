package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.koblas
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class QrBenchmark {
    @Param("4", "8", "12", "16", "32", "64", "128", "256", "512")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var square: F64DenseMatrix
    private lateinit var tall: F64DenseMatrix
    private lateinit var factored: F64QrDecomposition
    private lateinit var rhs: DoubleArray

    private lateinit var wide: F64QrDecomposition
    private lateinit var wideRhs: DoubleArray

    @Setup
    fun setup() {
        installDenseBackend(backend)
        val rng = benchRng()
        square = randomMatrix(n, n, rng)
        tall = randomMatrix(2 * n, n, rng)
        factored = tall.qr()
        rhs = randomVector(2 * n, rng)
        wide = randomMatrix(n, 2 * n, rng).transpose().qr()
        wideRhs = randomVector(n, rng)
    }

    @Benchmark
    fun qrSquare(): F64QrDecomposition = square.qr()

    @Benchmark
    fun qrTall(): F64QrDecomposition = tall.qr()

    @Benchmark
    fun qrPivotedSquare(): F64PivotedQrDecomposition = square.qrPivoted()

    @Benchmark
    fun qrPivotedTall(): F64PivotedQrDecomposition = tall.qrPivoted()

    @Benchmark
    fun leastSquares(): DoubleArray = factored.solve(rhs)

    @Benchmark
    fun applyQ(): DoubleArray = koblas.applyQ(factored, rhs)

    @Benchmark
    fun applyQTransposed(): DoubleArray = koblas.applyQ(factored, rhs, transpose = true)

    @Benchmark
    fun minimumNorm(): DoubleArray = koblas.solve(wide, wideRhs, minimumNorm = true)
}
