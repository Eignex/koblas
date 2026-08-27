package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import kotlinx.benchmark.*

/**
 * The sparse matrix products against the portable ones, which is what their dispatch gate is set from.
 *
 * `trsv` trails the three subjects alphabetically and no gate reaches it, so it is the control: a row from
 * the same run that the change under test cannot move.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseProductHostBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, FORCED_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var square: F64SparseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var dense: F64DenseMatrix
    private lateinit var product: F64DenseMatrix
    private lateinit var denseSingle: F64DenseMatrix
    private lateinit var productSingle: F64DenseMatrix
    private lateinit var triangle: F64SparseMatrix
    private lateinit var scratch: DoubleArray
    private lateinit var prepared: F64PreparedSparseMatrix

    @Setup
    fun setup() {
        installBackends(null)
        if (backend == FORCED_BACKEND) useUngatedSparseProduct()
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        square = a
        x = randomVector(n, rng)
        y = randomVector(n, rng)
        dense = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        product = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        denseSingle = randomMatrix(n, 1, rng)
        productSingle = randomMatrix(n, 1, rng)
        triangle = bandUpperTriangle(n)
        scratch = randomVector(n, rng)
        prepared = koblas.sparseBlas.prepare(a)
        println("resolved: sparseBlas=${koblas.sparseBlas.name} n=$n nnz(A)=${a.nnz}")
    }

    @TearDown
    fun tearDown() {
        prepared.close()
    }

    @Benchmark
    fun gemm(): F64DenseMatrix {
        koblas.sparseBlas.gemm(1.0, a, false, dense, false, 0.0, product)
        return product
    }

    /** The same native operation over one dense column, to expose whether its fixed marshalling cost pays. */
    @Benchmark
    fun gemmSingle(): F64DenseMatrix {
        koblas.sparseBlas.gemm(1.0, a, false, denseSingle, false, 0.0, productSingle)
        return productSingle
    }

    @Benchmark
    fun preparedGemm(): F64DenseMatrix {
        prepared.gemm(1.0, false, dense, 0.0, product)
        return product
    }

    @Benchmark
    fun preparedGemv(): DoubleArray {
        prepared.gemv(1.0, x, 0.0, y)
        return y
    }

    @Benchmark
    fun preparedSparseProduct(): F64SparseMatrix = prepared.gemm(square)

    @Benchmark
    fun gemv(): DoubleArray {
        koblas.sparseBlas.gemv(1.0, a, x, 0.0, y)
        return y
    }

    @Benchmark
    fun sparseProduct(): F64SparseMatrix = koblas.sparseBlas.gemm(a, square)

    @Benchmark
    fun trsv(): DoubleArray {
        x.copyInto(scratch)
        koblas.sparseBlas.trsv(triangle, scratch, lower = false)
        return scratch
    }
}

/** Right-hand sides for the level-3 row, enough that a multi-column call is not a single-column one. */
private const val RIGHT_HAND_SIDES = 8
