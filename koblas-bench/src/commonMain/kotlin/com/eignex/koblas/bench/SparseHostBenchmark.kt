package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param(BASIS_SHAPE, RANDOM_SHAPE)
    var shape: String = BASIS_SHAPE

    @Param(REFERENCE_BACKEND, AUTO_BACKEND, FORCED_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var factored: F64SparseFactorization

    @Setup
    fun setup() {
        installBackends(null)
        when (backend) {
            REFERENCE_BACKEND -> installBackends(koblas.with(sparseLu = F64ReferenceSparseLinearAlgebra))
            FORCED_BACKEND -> useUngatedSparseLu()
        }
        val rng = benchRng()
        a = if (shape == BASIS_SHAPE) simplexBasis(n, rng) else sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        factored = a.lu()
        println("resolved: sparseLu=${koblas.sparseLu.name} shape=$shape nnz(A)=${a.nnz} fill=${factored.nnz}")
    }

    @Benchmark
    fun factor(): F64SparseFactorization = a.lu()

    @Benchmark
    fun solve(): DoubleArray = factored.solve(rhs)

    @Benchmark
    fun solveTransposed(): DoubleArray = factored.solve(rhs, transpose = true)
}
