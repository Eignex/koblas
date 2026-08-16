package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.PivotedQrDecomposition
import com.eignex.koblas.dense.QrDecomposition
import com.eignex.koblas.dense.applyQ
import com.eignex.koblas.dense.qr
import com.eignex.koblas.dense.qrPivoted
import com.eignex.koblas.dense.solveLeastSquares
import com.eignex.koblas.dense.solveMinimumNorm
import com.eignex.koblas.koblas
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
 * Householder QR, whose column norms are the one place a factorization leans on `nrm2` rather than `dot`.
 * The tall shape is the least-squares case; the square one is where the norm work is largest relative to
 * the reflector updates.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class QrBenchmark {
    @Param("128", "256", "512")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var square: DenseMatrix
    private lateinit var tall: DenseMatrix
    private lateinit var factored: QrDecomposition
    private lateinit var rhs: DoubleArray

    /** A wide factorization for the minimum-norm path, which the tall shapes never exercise. */
    private lateinit var wide: QrDecomposition
    private lateinit var wideRhs: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        square = randomMatrix(n, n, rng)
        tall = randomMatrix(2 * n, n, rng)
        factored = tall.qr()
        rhs = randomVector(2 * n, rng)
        wide = randomMatrix(n, 2 * n, rng).qr()
        wideRhs = randomVector(n, rng)
    }

    @Benchmark
    fun qrSquare(): QrDecomposition = square.qr()

    @Benchmark
    fun qrTall(): QrDecomposition = tall.qr()

    @Benchmark
    fun qrPivotedSquare(): PivotedQrDecomposition = square.qrPivoted()

    @Benchmark
    fun qrPivotedTall(): PivotedQrDecomposition = tall.qrPivoted()

    @Benchmark
    fun leastSquares(): DoubleArray = factored.solveLeastSquares(rhs)

    /** Applying Q without forming it, the reflector sweep every QR solve is built on. */
    @Benchmark
    fun applyQ(): DoubleArray = koblas.applyQ(factored, rhs)

    @Benchmark
    fun applyQTransposed(): DoubleArray = koblas.applyQ(factored, rhs, transpose = true)

    /** The wide system's minimum-norm solve, which runs QR on the transpose. */
    @Benchmark
    fun minimumNorm(): DoubleArray = koblas.solveMinimumNorm(wide, wideRhs)
}
