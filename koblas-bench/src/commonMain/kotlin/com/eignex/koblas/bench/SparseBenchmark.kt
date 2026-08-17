package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.SparseLu
import com.eignex.koblas.sparse.cholesky
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.sparse.ldl
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.trsv
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray

    /** A symmetric operand for the LDL and Cholesky paths, plus a destination the triangular solve writes. */
    private lateinit var spd: SparseMatrix
    private lateinit var ldlFactored: SparseFactorization
    private lateinit var x: DoubleArray

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        spd = sparseSpdMatrix(n, rng)
        ldlFactored = spd.ldl()
        x = DoubleArray(n)
    }

    @Benchmark
    fun sparseLuSolve(): DoubleArray = a.lu(equilibrate = true).solve(rhs)

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = a.gemv(rhs, transpose = true)

    /** The factorization on its own, which the solve benchmark amortises away. */
    @Benchmark
    fun sparseLuFactor(): SparseFactorization = a.lu()

    @Benchmark
    fun sparseLuFactorEquilibrated(): SparseFactorization = a.lu(equilibrate = true)

    /** The symmetric paths, which take a different elimination from LU. */
    @Benchmark
    fun sparseLdlFactor(): SparseFactorization = spd.ldl()

    @Benchmark
    fun sparseCholeskyFactor(): SparseFactorization = spd.cholesky()

    @Benchmark
    fun sparseLdlSolve(): DoubleArray = ldlFactored.solve(rhs)

    /** Triangular solve straight off the CSC arrays, with no factorization in front of it. */
    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }
}
