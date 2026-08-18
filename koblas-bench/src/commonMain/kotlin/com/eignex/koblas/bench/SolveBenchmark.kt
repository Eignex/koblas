package com.eignex.koblas.bench

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.dense.F64LdlDecomposition
import com.eignex.koblas.dense.F64LuDecomposition
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.rcond
import com.eignex.koblas.dense.solve
import com.eignex.koblas.dense.trtri
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SolveBenchmark {
    @Param("16", "32", "64", "128", "256")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var a: F64DenseMatrix
    private lateinit var factored: F64LuDecomposition
    private lateinit var indefinite: F64LdlDecomposition
    private lateinit var rhs: DoubleArray
    private lateinit var x: DoubleArray

    /** A destination for the in-place refactorization, and the operands the added routines need. */
    private lateinit var reusable: F64LuDecomposition
    private lateinit var sym: F64DenseMatrix
    private lateinit var triangular: F64DenseMatrix
    private var anorm: Double = 0.0

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
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
    fun luFactor(): F64LuDecomposition = a.lu()

    @Benchmark
    fun luFactorInto(): F64LuDecomposition = koblas.factorInto(a, reusable)

    @Benchmark
    fun ldlFactor(): F64LdlDecomposition = koblas.ldl(sym)

    /** Hager's estimator, which iterates a solve against the factorization. */
    @Benchmark
    fun rcond(): Double = koblas.rcond(factored, anorm)

    /** Explicit inverse, which koblas documents as the thing to avoid; measured so the advice has a number. */
    @Benchmark
    fun invertLu(): F64DenseMatrix = koblas.invert(factored)

    @Benchmark
    fun trtri(): F64DenseMatrix = koblas.trtri(triangular, lower = true)
}
