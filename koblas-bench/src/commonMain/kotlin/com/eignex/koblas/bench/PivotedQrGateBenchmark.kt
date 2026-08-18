package com.eignex.koblas.bench

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.dense.F64PivotedQrDecomposition
import com.eignex.koblas.dense.qrPivoted
import kotlinx.benchmark.*

/**
 * Pivoted QR at the sizes [QrBenchmark] is too coarse to reach, so the crossover between a host `dgeqp3` and
 * the portable factorization can be found rather than assumed.
 *
 * Native dispatches LAPACK from size zero, on the reasoning that its portable kernels are scalar loops and a
 * foreign call is cheap there. Pivoted QR does more setup per call than the routines that threshold was
 * chosen for, so run this with the backend parameter crossed to see where the host actually starts winning.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class PivotedQrGateBenchmark {
    @Param("4", "8", "12", "16", "32", "64")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var square: F64DenseMatrix

    /** Tall as well as square, since a host routine's per-call cost is amortised differently by shape. */
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
