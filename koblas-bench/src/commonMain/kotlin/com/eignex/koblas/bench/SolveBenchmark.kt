package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SolveBenchmark {
    @Param("16", "32", "64", "128", "256")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64DenseMatrix
    private lateinit var rectangular: F64DenseMatrix
    private lateinit var factored: F64LuDecomposition
    private lateinit var indefinite: F64PivotedSymmetricIndefiniteDecomposition
    private lateinit var rhs: DoubleArray
    private lateinit var x: DoubleArray

    private lateinit var reusable: F64LuDecomposition
    private lateinit var sym: F64DenseMatrix
    private lateinit var triangular: F64DenseMatrix
    private var anorm: Double = 0.0

    @Setup
    fun setup() {
        installDenseBackend(backend)
        val rng = benchRng()
        a = dominantMatrix(n, rng)
        rectangular = F64DenseMatrix.zero(2 * n, n)
        for (j in 0 until n) for (i in 0 until 2 * n) {
            rectangular[i, j] = if (i == j) 2.0 else rng.nextDouble(-1.0, 1.0)
        }
        factored = a.lu()
        indefinite = koblas.pivotedSymmetricIndefinite(indefiniteMatrix(n, rng))
        rhs = randomVector(n, rng)
        x = DoubleArray(n)
        reusable = a.lu()
        sym = indefiniteMatrix(n, rng)
        triangular = dominantMatrix(n, rng)
        anorm = a.norm1()
    }

    @Benchmark
    fun luFactorAndSolve(): DoubleArray = a.lu().solve(rhs)

    @Benchmark
    fun luSolve(): DoubleArray = koblas.solve(factored, rhs)

    @Benchmark
    fun luSolveInto(): DoubleArray = koblas.solveInto(factored, rhs, x)

    @Benchmark
    fun ldlSolveInto(): DoubleArray = koblas.solveInto(indefinite, rhs, x)

    @Benchmark
    fun luFactor(): F64LuDecomposition = a.lu()

    @Benchmark
    fun rectangularLuFactor(): F64LuDecomposition = rectangular.lu()

    @Benchmark
    fun luFactorInto(): F64LuDecomposition = koblas.factorInto(a, reusable)

    @Benchmark
    fun pivotedSymmetricIndefiniteFactor(): F64PivotedSymmetricIndefiniteDecomposition =
        koblas.pivotedSymmetricIndefinite(sym)

    @Benchmark
    fun rcond(): Double = koblas.rcond(factored, anorm)

    @Benchmark
    fun invertLu(): F64DenseMatrix = koblas.invert(factored)

    @Benchmark
    fun trtri(): F64DenseMatrix = triangular.trtri(lower = true)
}
