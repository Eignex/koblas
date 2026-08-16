package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.LdlDecomposition
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.rcond
import com.eignex.koblas.dense.solve
import com.eignex.koblas.dense.trtri
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
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

    /** A destination for the in-place refactorization, and the operands the added routines need. */
    private lateinit var reusable: LuDecomposition
    private lateinit var sym: DenseMatrix
    private lateinit var triangular: DenseMatrix
    private var anorm: Double = 0.0

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        a = dominantMatrix(n, rng)
        factored = a.lu()
        indefinite = koblas.ldl(indefiniteMatrix(n, rng))
        rhs = randomVector(n, rng)
        x = DoubleArray(n)
        reusable = a.lu()
        sym = indefiniteMatrix(n, rng)
        triangular = dominantMatrix(n, rng)
        anorm = norm1(a)
    }

    @Benchmark
    fun luFactorAndSolve(): DoubleArray = a.lu().solve(rhs)

    @Benchmark
    fun luSolve(): DoubleArray = koblas.solve(factored, rhs)

    @Benchmark
    fun luSolveInto(): DoubleArray = koblas.solveInto(factored, rhs, x)

    @Benchmark
    fun ldlSolveInto(): DoubleArray = koblas.solveInto(indefinite, rhs, x)

    /** The factorizations themselves, which the solve benchmarks above amortise away. */
    @Benchmark
    fun luFactor(): LuDecomposition = a.lu()

    @Benchmark
    fun luFactorInto(): LuDecomposition = koblas.factorInto(a, reusable)

    @Benchmark
    fun ldlFactor(): LdlDecomposition = koblas.ldl(sym)

    /** Hager's estimator, which iterates a solve against the factorization. */
    @Benchmark
    fun rcond(): Double = koblas.rcond(factored, anorm)

    /** Explicit inverse, which koblas documents as the thing to avoid; measured so the advice has a number. */
    @Benchmark
    fun invertLu(): DenseMatrix = koblas.invert(factored)

    @Benchmark
    fun trtri(): DenseMatrix = koblas.trtri(triangular, lower = true)
}
