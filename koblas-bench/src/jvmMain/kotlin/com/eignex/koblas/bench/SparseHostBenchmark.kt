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

/**
 * The portable sparse LU against UMFPACK, on both the fixture that flatters neither and the one klause
 * actually factorizes.
 *
 * [shape] is the point of this suite. `random` is a matrix at one per cent density, which is where koblas's
 * sparse numbers came from until now and which is close to the worst case for any ordering — no structure,
 * fill toward dense, and a comparison that measures the fixture. `basis` is [simplexBasis], near-triangular
 * with slack columns and a few spikes, which is the shape a revised simplex hands a factorization. A gap that
 * appears on one and not the other is a fact about the input, not about the code, and that distinction is
 * what this exists to make.
 *
 * Factorization and solve are separate because a simplex refactorizes rarely and solves constantly: a backend
 * that factorizes slower and solves faster can still be the right one. JVM-only, since UMFPACK is bound there.
 * The resolved backend is printed from `@Setup`, because without SuiteSparse installed both parameter values
 * measure the same code and only that line says so.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param("basis", "random")
    var shape: String = "basis"

    /** `reference` pins the portable factorization; `auto` takes UMFPACK when discovery found it. */
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

    /** The refactorization cost on its own. */
    @Benchmark
    fun factor(): SparseFactorization = a.lu()

    /** One solve against factors built in `@Setup` — what a simplex iteration repeats. */
    @Benchmark
    fun solve(): DoubleArray = factored.solve(rhs)

    /** The transposed solve, a simplex's BTRAN. */
    @Benchmark
    fun solveTransposed(): DoubleArray = factored.solve(rhs, transpose = true)
}
