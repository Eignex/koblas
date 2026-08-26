package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

/**
 * The two symmetric factorizations against the portable ones, which is what their dispatch gate is set from.
 *
 * `transpose` trails the two subjects alphabetically and no gate reaches it, so it is the control: a row
 * from the same run that the change under test cannot move.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseSymmetricHostBenchmark {
    @Param("128", "192", "256", "1024")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, FORCED_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix

    @Setup
    fun setup() {
        installBackends(null)
        when (backend) {
            REFERENCE_BACKEND -> installBackends(koblas.with(sparseDecompositions = F64ReferenceSparseLinearAlgebra))
            FORCED_BACKEND -> useUngatedSparseLu()
        }
        val rng = benchRng()
        a = sparseSpdMatrix(n, rng)
        val half = koblas.sparseDecompositions.name
        println("resolved: sparseDecompositions=$half n=$n nnz(A)=${a.nnz}")
    }

    @Benchmark
    fun cholesky(): F64SparseFactorization = a.cholesky()

    @Benchmark
    fun ldl(): F64SparseFactorization = a.ldl()

    @Benchmark
    fun transpose(): F64SparseMatrix = a.transpose()
}
