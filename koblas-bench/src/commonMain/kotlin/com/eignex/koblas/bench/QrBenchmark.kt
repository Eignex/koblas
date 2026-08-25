package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.koblas
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class QrBenchmark {
    @Param("128", "256", "512")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var square: F64DenseMatrix
    private lateinit var tall: F64DenseMatrix
    private lateinit var factored: F64QrDecomposition
    private lateinit var rhs: DoubleArray

    private lateinit var wide: F64QrDecomposition
    private lateinit var wideRhs: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        square = randomMatrix(n, n, rng)
        tall = randomMatrix(2 * n, n, rng)
        factored = tall.qr()
        rhs = randomVector(2 * n, rng)
        wide = randomMatrix(n, 2 * n, rng).qr()
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
    fun leastSquares(): DoubleArray = factored.solveLeastSquares(rhs)

    @Benchmark
    fun applyQ(): DoubleArray = koblas.applyQ(factored, rhs)

    @Benchmark
    fun applyQTransposed(): DoubleArray = koblas.applyQ(factored, rhs, transpose = true)

    @Benchmark
    fun minimumNorm(): DoubleArray = koblas.solveMinimumNorm(wide, wideRhs)
}
