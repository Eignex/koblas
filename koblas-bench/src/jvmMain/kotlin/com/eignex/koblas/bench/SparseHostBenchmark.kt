package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.lu
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

/** Without SuiteSparse installed both backend values measure the same code, and only the resolved line says so. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param("basis", "random")
    var shape: String = "basis"

    /** reference pins the portable factorization, auto takes UMFPACK when discovery found it. */
    @Param("reference", "auto")
    var backend: String = "reference"

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var factored: SparseFactorization

    @Setup
    fun setup() {
        val portable = koblas.with(sparseLapack = ReferenceSparseLinearAlgebra)
        installBackends(if (backend == REFERENCE_BACKEND) portable else null)
        val rng = Random(BENCH_SEED)
        a = if (shape == "basis") simplexBasis(n, rng) else sparseDominantMatrix(n, density = 0.01, rng = rng)
        rhs = randomVector(n, rng)
        factored = a.lu()
        println("resolved: sparseLapack=${koblas.sparseLapack.name} shape=$shape nnz(A)=${a.nnz} fill=${factored.nnz}")
    }

    @Benchmark
    fun factor(): SparseFactorization = a.lu()

    @Benchmark
    fun solve(): DoubleArray = factored.solve(rhs)

    @Benchmark
    fun solveTransposed(): DoubleArray = factored.solve(rhs, transpose = true)
}
