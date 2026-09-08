package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*
import kotlinx.benchmark.*

/**
 * The sparse matrix products against the portable ones.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseProductHostBenchmark {
    @Param("31", "32", "33", "63", "64", "65", "255", "256", "257", "1024")
    var n: Int = 0

    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND)
    var sparseArm: String = BUILTIN_BACKEND

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    @Param("regular", "banded", "skewed")
    var shape: String = "regular"

    private var builtIn: F64SparseBlas? = null
    private var external: SparseComparator? = null

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
    private lateinit var triangularProduct: F64DenseMatrix
    private lateinit var triangularSolve: F64DenseMatrix
    private lateinit var triangularProductRight: F64DenseMatrix
    private lateinit var prepared: F64PreparedSparseMatrix
    private var externalPrepared: PreparedSparseComparator? = null
    private var externalTriangle: PreparedSparseComparator? = null

    @Setup
    fun setup() {
        val rng = benchRng()
        if (sparseArm == BUILTIN_BACKEND) {
            val context = explicitBuiltInContext()
            builtIn = context.sparseBlas
            check(context.sparseBlas.name == REFERENCE_BACKEND) { "built-in sparse arm resolved ${context.sparseBlas.name}" }
            println("resolved: arm=$sparseArm sparse=${context.sparseBlas.name}/${context.sparseKernels.name} threading=single calling thread")
        } else {
            external = checkNotNull(oneMklSparseComparator()) { "the benchmark-only oneMKL sparse comparator is unavailable" }
            check(external!!.identity == "onemkl/sparse-blas" && external!!.threading == "1 thread")
            println("resolved: arm=$sparseArm sparse=${external!!.identity} threading=${external!!.threading}")
        }
        a = sparseComparisonMatrix(n + 1, n - 1, density, shape, rng)
        square = sparseComparisonMatrix(n - 1, n + 2, density, shape, rng)
        x = randomVector(n - 1, rng)
        y = randomVector(n + 1, rng)
        dense = randomMatrix(n - 1, RIGHT_HAND_SIDES, rng)
        product = randomMatrix(n + 1, RIGHT_HAND_SIDES, rng)
        denseSingle = randomMatrix(n - 1, 1, rng)
        productSingle = randomMatrix(n + 1, 1, rng)
        triangle = bandUpperTriangle(n)
        scratch = randomVector(n, rng)
        triangularProduct = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        triangularSolve = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        triangularProductRight = randomMatrix(RIGHT_HAND_SIDES, n, rng)
        if (builtIn != null) prepared = builtIn!!.prepare(a) else externalPrepared = external!!.prepare(a)
        externalTriangle = external?.prepare(triangle, triangular = true, lower = false)
        println("workload: n=$n density=$density shape=$shape nnz(A)=${a.nnz}")
        reportAllocatingWorkload("sparse/$sparseArm/product", "fresh sparse result and output-pattern construction")
    }

    @TearDown
    fun tearDown() {
        if (::prepared.isInitialized) prepared.close()
        externalPrepared?.close()
        externalTriangle?.close()
    }

    @Benchmark
    fun gemm(): F64DenseMatrix {
        if (external != null) {
            external!!.prepare(a).use { it.gemm(1.0, dense, 0.0, product) }
        } else {
            builtIn!!.gemm(1.0, a, false, dense, false, 0.0, product)
        }
        return product
    }

    /** The same native operation over one dense column, to expose whether its fixed marshalling cost pays. */
    @Benchmark
    fun gemmSingle(): F64DenseMatrix {
        if (external != null) {
            external!!.prepare(a).use { it.gemm(1.0, denseSingle, 0.0, productSingle) }
        } else {
            builtIn!!.gemm(1.0, a, false, denseSingle, false, 0.0, productSingle)
        }
        return productSingle
    }

    @Benchmark
    fun preparedGemm(): F64DenseMatrix {
        externalPrepared?.gemm(1.0, dense, 0.0, product) ?: prepared.gemm(1.0, false, dense, 0.0, product)
        return product
    }

    @Benchmark
    fun preparedGemv(): DoubleArray {
        externalPrepared?.gemv(1.0, x, 0.0, y) ?: prepared.gemv(1.0, x, 0.0, y)
        return y
    }

    @Benchmark
    fun preparedSparseProduct(): F64SparseMatrix = external?.sparseProduct(a, square) ?: prepared.gemm(square)

    @Benchmark
    fun gemv(): DoubleArray {
        if (external != null) {
            external!!.prepare(a).use { it.gemv(1.0, x, 0.0, y) }
        } else {
            builtIn!!.gemv(1.0, a, x, 0.0, y)
        }
        return y
    }

    @Benchmark
    fun sparseProduct(): F64SparseMatrix = external?.sparseProduct(a, square) ?: builtIn!!.gemm(a, square)

    @Benchmark
    fun trsv(): DoubleArray {
        x.copyInto(scratch)
        val comparator = externalTriangle
        if (comparator != null) comparator.trsv(x, scratch) else builtIn!!.trsv(triangle, scratch, lower = false)
        return scratch
    }

    @Benchmark
    fun trmv(): DoubleArray {
        x.copyInto(scratch)
        val comparator = externalTriangle
        if (comparator != null) comparator.trmv(x, scratch) else builtIn!!.trmv(triangle, scratch, lower = false)
        return scratch
    }

    @Benchmark
    fun trmm(): F64DenseMatrix {
        dense.data.copyInto(triangularProduct.data)
        val comparator = externalTriangle
        if (comparator != null) comparator.trmm(dense, triangularProduct) else builtIn!!.trmm(triangle, triangularProduct, lower = false)
        return triangularProduct
    }

    @Benchmark
    fun trsm(): F64DenseMatrix {
        val comparator = externalTriangle
        if (comparator != null) {
            comparator.trsm(dense, triangularSolve)
        } else {
            dense.data.copyInto(triangularSolve.data)
            builtIn!!.trsm(triangle, triangularSolve, lower = false)
        }
        return triangularSolve
    }

    @Benchmark
    fun trmmRight(): F64DenseMatrix {
        triangularProductRight.data.fill(1.0)
        if (external != null) {
            // oneMKL has no right-side sparse triangular multiply. This row is intentionally a composition
            // and is excluded from direct parity: B*A = transpose(A^T*transpose(B)).
            val transposedInput = F64DenseMatrix.wrap(n, RIGHT_HAND_SIDES, DoubleArray(n * RIGHT_HAND_SIDES))
            for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) transposedInput[j, i] = triangularProductRight[i, j]
            val transposedOut = F64DenseMatrix.zero(n, RIGHT_HAND_SIDES)
            externalTriangle!!.trmm(transposedInput, transposedOut, transpose = true)
            for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) triangularProductRight[i, j] = transposedOut[j, i]
        } else {
            builtIn!!.trmm(triangle, triangularProductRight, lower = false, right = true)
        }
        return triangularProductRight
    }
}

/** Right-hand sides for the level-3 row, enough that a multi-column call is not a single-column one. */
private const val RIGHT_HAND_SIDES = 8
