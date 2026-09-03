package com.eignex.koblas.bench

import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64CholeskyDecomposition
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.rankUpdate
import kotlinx.benchmark.*

/**
 * Cholesky rank updates on either side of the host half's rank gate, which is what a report has to show to
 * justify where that gate sits: the portable sweep costs one pass over the triangle per vector, while
 * `dtpqrt` pays a transpose once and then blocks.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class CholeskyUpdateBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param("1", "8", "16", "32", "64")
    var rank: Int = 0

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var factor: F64CholeskyDecomposition
    private lateinit var block: F64DenseMatrix
    private lateinit var vector: DoubleArray
    private lateinit var workspace: Workspace

    @Setup
    fun setup() {
        installDenseBackend(backend)
        val rng = benchRng()
        factor = spdMatrix(n, rng).cholesky()
        // The update runs in place with no per-call reset, and every update adds mass to the factor, so the
        // vectors are scaled far down: at this size a whole trial moves the factor by less than a rounding
        // error, where unit-scale vectors would run the diagonal up toward Infinity partway through.
        val scale = 1e-8
        block = F64DenseMatrix.wrap(n, rank, DoubleArray(n * rank) { scale * rng.nextDouble(-1.0, 1.0) })
        vector = DoubleArray(n) { scale * rng.nextDouble(-1.0, 1.0) }
        workspace = Workspace()
    }

    @Benchmark
    fun rankUpdate(): F64CholeskyDecomposition = factor.rankUpdate(vector, workspace = workspace)

    @Benchmark
    fun rankUpdateBlock(): F64CholeskyDecomposition = factor.rankUpdate(block, workspace = workspace)
}
