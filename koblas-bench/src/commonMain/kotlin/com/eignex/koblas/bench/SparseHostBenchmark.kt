package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param(BASIS_SHAPE, RANDOM_SHAPE)
    var shape: String = BASIS_SHAPE

    @Param(AUTOMATIC_BACKEND, REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var factored: F64SparseFactorization

    @Setup
    fun setup() {
        installSparseDecompositionBackend(backend)
        val rng = benchRng()
        a = if (shape == BASIS_SHAPE) simplexBasis(n, rng) else sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        factored = a.lu()
        println("resolved: sparseDecompositions=${koblas.sparseDecompositions.name} shape=$shape nnz(A)=${a.nnz} fill=${factored.nnz}")
    }

    @Benchmark
    fun factor(): F64SparseFactorization = a.lu()

    // Reading the factors copies them out of the library, so it is its own row rather than part of [factor].
    @Benchmark
    fun factors(): Int = a.lu().use { it.l.nnz + it.u.nnz }

    @Benchmark
    fun solve(): DoubleArray = factored.solve(rhs)

    @Benchmark
    fun solveTransposed(): DoubleArray = factored.solve(rhs, transpose = true)

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}
