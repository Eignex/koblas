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
 * The portable sparse LU against SuiteSparse's UMFPACK, which is the yardstick koblas's own factorization
 * has not had one of until now.
 *
 * JVM-only because UMFPACK is bound on the JVM only so far. `auto` needs a machine with SuiteSparse
 * installed; without it discovery registers nothing and both parameter values measure the same kernels, so
 * the resolved line printed from `@Setup` is what says whether a run means anything.
 *
 * Factorization and solve are separate benchmarks rather than one paired measurement, because the two
 * answer different questions. A simplex refactorizes rarely and solves constantly, so a backend that
 * factorizes slower and solves faster can still be the right one — [SparseBenchmark] measures the pair the
 * way a refactorization pays it, and this separates them.
 *
 * [equilibrate] is off here: asking for it sends the UMFPACK backend down the portable path by design, so
 * leaving it on would compare the reference against itself.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseHostBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    /** `reference` pins the portable factorization; `auto` takes UMFPACK when discovery found it. */
    @Param("reference", "auto")
    var backend: String = "reference"

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var factored: SparseFactorization

    @Setup
    fun setup() {
        // null leaves the installed slot empty, so discovery's registration stands — UMFPACK when the
        // machine has SuiteSparse, the portable factorization when it does not.
        val portable = koblas.with(sparseLapack = ReferenceSparseLinearAlgebra)
        installBackends(if (backend == REFERENCE_BACKEND) portable else null)
        println("resolved: sparseLapack=${koblas.sparseLapack.name}")
        val rng = Random(BENCH_SEED)
        a = sparseDominantMatrix(n, density = 0.01, rng = rng)
        rhs = randomVector(n, rng)
        factored = a.lu()
        println("fill: nnz=${factored.nnz}, class=${factored::class.simpleName}")
    }

    /** The refactorization cost on its own. */
    @Benchmark
    fun factor(): SparseFactorization = a.lu()

    /** One solve against factors built in `@Setup`, which is the operation a simplex iteration repeats. */
    @Benchmark
    fun solve(): DoubleArray = factored.solve(rhs)

    /** The transposed solve, a simplex's BTRAN; UMFPACK takes it through its own `sys` selector. */
    @Benchmark
    fun solveTransposed(): DoubleArray = factored.solve(rhs, transpose = true)
}
