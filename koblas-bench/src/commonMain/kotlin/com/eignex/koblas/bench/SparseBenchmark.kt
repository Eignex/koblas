package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
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

    private lateinit var luFactored: F64SparseFactorization

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        luFactored = a.lu()
        x = DoubleArray(n)
    }

    @Benchmark
    fun sparseLuSolve(): DoubleArray = a.lu(equilibrate = true).solve(rhs)

    @Benchmark
    fun sparseLuFtran(): DoubleArray = luFactored.solveInto(rhs, x)

    @Benchmark
    fun sparseLuBtran(): DoubleArray = luFactored.solveInto(rhs, x, transpose = true)

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = a.gemv(rhs, transpose = true)

    @Benchmark
    fun sparseLuFactor(): F64SparseFactorization = a.lu()

    @Benchmark
    fun sparseLuFactorEquilibrated(): F64SparseFactorization = a.lu(equilibrate = true)

    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }

    @Benchmark
    fun sparseTranspose(): F64SparseMatrix = a.transpose()
}
