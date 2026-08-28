package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
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

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var b: DoubleArray
    private lateinit var block: F64DenseMatrix
    private lateinit var factored: F64SparseQrFactorization

    @Setup
    fun setup() {
        installBackends(null)
        when (backend) {
            REFERENCE_BACKEND -> installBackends(koblas.with(sparseDecompositions = F64ReferenceSparseLinearAlgebra))
            HOST_BACKEND -> useSparseLu()
        }
        val rng = benchRng()
        a = sparseTallMatrix(2 * n, n, rng)
        b = DoubleArray(2 * n) { rng.nextDouble(-1.0, 1.0) }
        block = randomMatrix(2 * n, QR_BLOCK_COLUMNS, rng)
        factored = a.qr()
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half rows=${a.rows} cols=${a.cols} nnz(A)=${a.nnz}")
    }

    // Returned rather than reduced to a field, so the factorization escapes and the JIT cannot discard it,
    // and closed rather than left to the cleaner, which would measure the collector. Reading `r` does both
    // too, and charges a native binding a second factorization the portable one never performs.
    @Benchmark
    fun qr(): F64SparseQrFactorization = a.qr().also { it.close() }

    @Benchmark
    fun qrApplyQ(): DoubleArray = factored.applyQ(b, transpose = true)

    @Benchmark
    fun qrBlockSolve(): F64DenseMatrix = factored.solve(block)

    // Separate from [qr] because a native binding materialises the factors through a second factorization,
    // which a combined row would charge to the first.
    @Benchmark
    fun qrFactors(): Int = a.qr().use { it.r.nnz + it.rank }

    @Benchmark
    fun qrSolve(): DoubleArray = a.qr().use { it.solve(b) }

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}

/** Right-hand sides in the blocked solve, matching the count the other block suites use. */
private const val QR_BLOCK_COLUMNS = 8
