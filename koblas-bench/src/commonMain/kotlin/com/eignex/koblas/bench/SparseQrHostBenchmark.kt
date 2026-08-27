package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

/** The sparse QR against the portable one, with `transpose` as the control no gate reaches. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseQrHostBenchmark {
    @Param("128", "256", "512", "1024")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, FORCED_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var b: DoubleArray

    @Setup
    fun setup() {
        installBackends(null)
        when (backend) {
            REFERENCE_BACKEND -> installBackends(koblas.with(sparseDecompositions = F64ReferenceSparseLinearAlgebra))
            FORCED_BACKEND -> useUngatedSparseLu()
        }
        val rng = benchRng()
        a = sparseTallMatrix(2 * n, n, rng)
        b = DoubleArray(2 * n) { rng.nextDouble(-1.0, 1.0) }
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half rows=${a.rows} cols=${a.cols} nnz(A)=${a.nnz}")
    }

    // Closed rather than left to the cleaner, which would measure the collector too.
    @Benchmark
    fun qr(): Int = a.qr().use { it.n }

    @Benchmark
    fun qrSolve(): DoubleArray = a.qr().use { it.solve(b) }

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}
