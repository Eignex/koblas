package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.lu
import kotlinx.benchmark.*

/** Without SuiteSparse installed both backend values measure the same code, and only the resolved line says so. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param(BASIS_SHAPE, RANDOM_SHAPE)
    var shape: String = BASIS_SHAPE

    /** [REFERENCE_BACKEND] pins the portable factorization, [AUTO_BACKEND] takes UMFPACK when discovery found it. */
    @Param(REFERENCE_BACKEND, AUTO_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var factored: F64SparseFactorization

    @Setup
    fun setup() {
        val portable = koblas.with(sparseLu = F64ReferenceSparseLinearAlgebra)
        installBackends(if (backend == REFERENCE_BACKEND) portable else null)
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
