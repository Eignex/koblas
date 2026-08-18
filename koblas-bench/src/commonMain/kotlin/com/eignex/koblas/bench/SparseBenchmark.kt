package com.eignex.koblas.bench

import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.cholesky
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.sparse.ldl
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.trsv
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

    /** A symmetric operand for the LDL and Cholesky paths, plus a destination the triangular solve writes. */
    private lateinit var spd: F64SparseMatrix
    private lateinit var ldlFactored: F64SparseFactorization
    private lateinit var x: DoubleArray

    /** Factored once in setup, so the solve benchmarks below time the sweeps and not the elimination. */
    private lateinit var luFactored: F64SparseFactorization

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        spd = sparseSpdMatrix(n, rng)
        ldlFactored = spd.ldl()
        luFactored = a.lu()
        x = DoubleArray(n)
    }

    @Benchmark
    fun sparseLuSolve(): DoubleArray = a.lu(equilibrate = true).solve(rhs)

    /** The forward sweeps alone, `L U (Qᵀx) = P(Eb)`, against a factorization setup already paid for. */
    @Benchmark
    fun sparseLuFtran(): DoubleArray = luFactored.solveInto(rhs, x)

    /** The transposed sweeps alone, which walk the factors by column where [sparseLuFtran] walks them by row. */
    @Benchmark
    fun sparseLuBtran(): DoubleArray = luFactored.solveInto(rhs, x, transpose = true)

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = a.gemv(rhs, transpose = true)

    /** The factorization on its own, which the solve benchmark amortises away. */
    @Benchmark
    fun sparseLuFactor(): F64SparseFactorization = a.lu()

    @Benchmark
    fun sparseLuFactorEquilibrated(): F64SparseFactorization = a.lu(equilibrate = true)

    /** The symmetric paths, which take a different elimination from LU. */
    @Benchmark
    fun sparseLdlFactor(): F64SparseFactorization = spd.ldl()

    @Benchmark
    fun sparseCholeskyFactor(): F64SparseFactorization = spd.cholesky()

    @Benchmark
    fun sparseLdlSolve(): DoubleArray = ldlFactored.solve(rhs)

    /** Triangular solve straight off the CSC arrays, with no factorization in front of it. */
    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }

    /** Costs by stored entry rather than by dimension, so it scales with the fill and not with `n`. */
    @Benchmark
    fun sparseTranspose(): F64SparseMatrix = a.transpose()
}
