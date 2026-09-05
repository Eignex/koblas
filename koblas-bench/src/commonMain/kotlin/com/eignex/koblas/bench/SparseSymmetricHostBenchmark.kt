package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

/**
 * The two symmetric factorizations against the portable ones.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseSymmetricHostBenchmark {
    @Param("128", "192", "256", "1024")
    var n: Int = 0

    @Param(AUTOMATIC_BACKEND, REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var choleskyFactor: F64SparseCholeskyFactorization
    private lateinit var rhs: DoubleArray
    private lateinit var solution: DoubleArray
    private lateinit var choleskyAnalysis: F64SparseSymbolicAnalysis<F64SparseCholeskyFactorization>
    private lateinit var ldlAnalysis: F64SparseSymbolicAnalysis<F64QuasiDefiniteLdlFactorization>

    @Setup
    fun setup() {
        installSparseDecompositionBackend(backend)
        val rng = benchRng()
        a = sparseSpdMatrix(n, rng)
        choleskyFactor = a.cholesky()
        rhs = DoubleArray(n) { it * 0.125 - 1.0 }
        solution = DoubleArray(n)
        choleskyAnalysis = koblas.sparseCholesky.analyzeCholesky(a)
        ldlAnalysis = koblas.quasiDefiniteLdl.analyzeQuasiDefiniteLdl(a)
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half n=$n nnz(A)=${a.nnz}")
    }

    @TearDown
    fun tearDown() {
        choleskyFactor.close()
        choleskyAnalysis.close()
        ldlAnalysis.close()
    }

    @Benchmark
    fun cholesky(): F64SparseFactorization = a.cholesky()

    @Benchmark
    fun quasiDefiniteLdl(): F64SparseFactorization = a.quasiDefiniteLdl()

    // The solve against a factor built once, which is where a repeated right-hand side spends its time and
    // where a binding holding a native descriptor across calls is measured.
    @Benchmark
    fun choleskySolve(): DoubleArray = choleskyFactor.solveInto(rhs, solution)

    // Against the two above: the same numeric sweep with the pattern already analyzed, so the pair measures
    // what a caller refactorizing one structure saves.
    @Benchmark
    fun choleskyRefactor(): F64SparseFactorization = choleskyAnalysis.factor(a)

    @Benchmark
    fun quasiDefiniteLdlRefactor(): F64SparseFactorization = ldlAnalysis.factor(a)

    // Reading the factors converts a copy of the factor, so it is its own row rather than part of the two
    // above. `choleskyFactors` and `quasiDefiniteLdlFactors` sort beside their factorizations, before the control.
    @Benchmark
    fun choleskyFactors(): Int = a.cholesky().use { it.l.nnz }

    @Benchmark
    fun quasiDefiniteLdlFactors(): Int = a.quasiDefiniteLdl().use { it.l.nnz + it.d.size }

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}
