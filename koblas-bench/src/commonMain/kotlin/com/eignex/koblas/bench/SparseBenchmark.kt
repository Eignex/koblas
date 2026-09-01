package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: F64SparseMatrix
    private lateinit var rhs: DoubleArray

    private lateinit var x: DoubleArray
    private lateinit var multiplied: DoubleArray

    private lateinit var luFactored: F64SparseFactorization

    /** The portable factorization set to scale rows, which is where equilibration lives now. */
    private val equilibrating = F64ReferenceSparseDecompositions(equilibrate = true)

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        luFactored = a.lu()
        x = DoubleArray(n)
        multiplied = DoubleArray(n)
    }

    @Benchmark
    fun sparseLuSolve(): DoubleArray = equilibrating.factor(a).solve(rhs)

    @Benchmark
    fun sparseLuFtran(): DoubleArray = luFactored.solveInto(rhs, x)

    @Benchmark
    fun sparseLuBtran(): DoubleArray = luFactored.solveInto(rhs, x, transpose = true)

    @Benchmark
    fun sparseGemv(): DoubleArray = koblas.gemv(a, rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = koblas.gemv(a, rhs, transpose = true)

    @Benchmark
    fun sparseLuFactor(): F64SparseFactorization = a.lu()

    @Benchmark
    fun sparseLuFactorEquilibrated(): F64SparseFactorization =
        equilibrating.factor(a)

    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }

    @Benchmark
    fun sparseTrmv(): DoubleArray {
        rhs.copyInto(multiplied)
        a.trmv(multiplied, lower = true)
        return multiplied
    }

    @Benchmark
    fun sparseTranspose(): F64SparseMatrix = a.transpose()
}
