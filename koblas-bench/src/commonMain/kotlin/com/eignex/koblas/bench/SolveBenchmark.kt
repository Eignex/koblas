package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.koblas
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve
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
 * Factorization and single right-hand-side solves, dense general and symmetric indefinite.
 *
 * Factorization is `O(n^3)` and stays on the native backend, while the triangular solves that follow it
 * are `O(n^2)` and lose to the portable kernels below roughly `n = 1024` — the same shape as level 2.
 * Both are here so a change that moves the crossover shows up against the factorization it is bundled
 * with in practice. The `into` variants measure the same shapes as the allocating ones, which is what
 * makes the destination-passing saving visible rather than inferred.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SolveBenchmark {
    @Param("16", "32", "64", "128", "256")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var a: DenseMatrix
    private lateinit var factored: LuDecomposition
    private lateinit var indefinite: LdlDecomposition
    private lateinit var rhs: DoubleArray
    private lateinit var x: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        a = dominantMatrix(n, rng)
        factored = a.lu()
        indefinite = koblas.ldl(indefiniteMatrix(n, rng))
        rhs = randomVector(n, rng)
        x = DoubleArray(n)
    }

    /** LINPACK-style dense solve: factorize, then forward and back substitute. */
    @Benchmark
    fun luFactorAndSolve(): DoubleArray = a.lu().solve(rhs)

    @Benchmark
    fun luSolve(): DoubleArray = koblas.solve(factored, rhs)

    @Benchmark
    fun luSolveInto(): DoubleArray = koblas.solveInto(factored, rhs, x)

    @Benchmark
    fun ldlSolveInto(): DoubleArray = koblas.solveInto(indefinite, rhs, x)
}
