package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

/**
 * The two symmetric factorizations against the portable ones.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseSymmetricHostBenchmark {
    @Param("128", "192", "256", "1024")
    var n: Int = 0

    @Param(AUTOMATIC_BACKEND, REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix

    @Setup
    fun setup() {
        installSparseDecompositionBackend(backend)
        val rng = benchRng()
        a = sparseSpdMatrix(n, rng)
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half n=$n nnz(A)=${a.nnz}")
    }

    @Benchmark
    fun cholesky(): F64SparseFactorization = a.cholesky()

    @Benchmark
    fun ldl(): F64SparseFactorization = a.ldl()

    // Reading the factors converts a copy of the factor, so it is its own row rather than part of the two
    // above. `choleskyFactors` and `ldlFactors` sort beside their factorizations, before the control.
    @Benchmark
    fun choleskyFactors(): Int = a.cholesky().use { it.l.nnz }

    @Benchmark
    fun ldlFactors(): Int = a.ldl().use { it.l.nnz + it.d.size }

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}
