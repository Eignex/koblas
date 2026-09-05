package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import kotlinx.benchmark.*

/** Entries below the diagonal of each banded column, few enough that the factor takes almost no fill. */
internal const val REFACTOR_BANDWIDTH = 2

/**
 * A symbolic analysis reused across numeric refactorizations, against factorizing from scratch.
 *
 * The matrix is banded rather than uniformly sparse: what the analysis saves is the elimination tree and the
 * column counts, which cost a pass over the pattern whatever the fill, so the saving shows only where the
 * numeric sweep is not doing far more work than that. [SparseSymmetricHostBenchmark] carries the same pair
 * over the filled pattern, where it is not.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseRefactorBenchmark {
    @Param("256", "1024", "4096")
    var n: Int = 0

    @Param(AUTOMATIC_BACKEND, REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var analysis: F64SparseSymbolicAnalysis<F64SparseCholeskyFactorization>

    @Setup
    fun setup() {
        installSparseDecompositionBackend(backend)
        a = sparseBandedSpdMatrix(n, REFACTOR_BANDWIDTH, benchRng())
        analysis = koblas.sparseCholesky.analyzeCholesky(a)
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half n=$n nnz(A)=${a.nnz}")
    }

    @TearDown
    fun tearDown() {
        analysis.close()
    }

    @Benchmark
    fun bandedCholesky(): F64SparseCholeskyFactorization = a.cholesky().also { it.close() }

    @Benchmark
    fun bandedCholeskyRefactor(): F64SparseCholeskyFactorization = analysis.factor(a).also { it.close() }
}
