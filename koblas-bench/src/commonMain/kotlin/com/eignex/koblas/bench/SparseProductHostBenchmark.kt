package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
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
    var productShape: String = "regular"

    var triangleVariant: String = "upper-nontrans-nonunit"

    private lateinit var arm: SparseBenchmarkArm
    private lateinit var resources: BenchmarkResources

    private lateinit var a: SparseMatrix
    private lateinit var square: SparseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var dense: DenseMatrix
    private lateinit var product: DenseMatrix
    private lateinit var denseSingle: DenseMatrix
    private lateinit var productSingle: DenseMatrix
    private lateinit var triangle: SparseMatrix
    private lateinit var triangularVector: DoubleArray
    private lateinit var triangularDense: DenseMatrix
    private lateinit var triangularDenseRight: DenseMatrix
    private lateinit var scratch: DoubleArray
    private lateinit var triangularProduct: DenseMatrix
    private lateinit var triangularSolve: DenseMatrix
    private lateinit var triangularProductRight: DenseMatrix
    private lateinit var prepared: PreparedSparseBenchmarkArm
    private lateinit var preparedSquare: PreparedSparseBenchmarkArm
    private lateinit var preparedTriangle: PreparedSparseBenchmarkArm
    private var triangleLower: Boolean = false
    private var triangleTranspose: Boolean = false
    private var triangleUnitDiag: Boolean = false

    @Setup
    fun setup() {
        val rng = benchRng()
        arm = SparseBenchmarkArm.resolve(sparseArm)
        resources = BenchmarkResources()
        a = sparseComparisonMatrix(n + 1, n - 1, density, productShape, rng)
        square = sparseComparisonMatrix(n - 1, n + 2, density, productShape, rng)
        x = randomVector(n - 1, rng)
        y = randomVector(n + 1, rng)
        dense = randomMatrix(n - 1, RIGHT_HAND_SIDES, rng)
        product = randomMatrix(n + 1, RIGHT_HAND_SIDES, rng)
        denseSingle = randomMatrix(n - 1, 1, rng)
        productSingle = randomMatrix(n + 1, 1, rng)
        val triangleParts = triangleVariant.split('-')
        check(triangleParts.size == 3) { "unknown triangular variant: $triangleVariant" }
        triangleLower = when (triangleParts[0]) {
            "lower" -> true
            "upper" -> false
            else -> error("unknown triangle: ${triangleParts[0]}")
        }
        triangleTranspose = when (triangleParts[1]) {
            "trans" -> true
            "nontrans" -> false
            else -> error("unknown transpose: ${triangleParts[1]}")
        }
        triangleUnitDiag = when (triangleParts[2]) {
            "unit" -> true
            "nonunit" -> false
            else -> error("unknown diagonal: ${triangleParts[2]}")
        }
        triangle = bandTriangle(n, triangleLower)
        triangularVector = randomVector(n, rng)
        triangularDense = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        triangularDenseRight = randomMatrix(RIGHT_HAND_SIDES, n, rng)
        scratch = DoubleArray(n)
        triangularProduct = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
        triangularSolve = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
        triangularProductRight = DenseMatrix.zero(RIGHT_HAND_SIDES, n)
        prepared = resources.acquire { arm.prepare(a) }
        preparedSquare = resources.acquire { arm.prepareProductRight(square) }
        preparedTriangle = resources.acquire {
            arm.prepare(
                triangle,
                SparseDescriptor(triangular = true, lower = triangleLower, unitDiag = triangleUnitDiag),
            )
        }
        println("workload: n=$n density=$density shape=$productShape nnz(A)=${a.nnz}")
        reportAllocatingWorkload(
            "sparse/$sparseArm/prepared-product",
            "reusable operand preparation is outside the measured row; fresh result construction remains timed",
        )
        reportAllocatingWorkload(
            "sparse/$sparseArm/one-shot-product",
            "operand preparation or conversion plus fresh sparse result construction",
        )
    }

    @TearDown
    fun tearDown() {
        resources.close()
    }

    @Benchmark
    fun gemm(): DenseMatrix {
        arm.gemm(1.0, a, dense, 0.0, product)
        return product
    }

    /** The same native operation over one dense column, to expose whether its fixed marshalling cost pays. */
    @Benchmark
    fun gemmSingle(): DenseMatrix {
        arm.gemm(1.0, a, denseSingle, 0.0, productSingle)
        return productSingle
    }

    @Benchmark
    fun preparedGemm(): DenseMatrix {
        prepared.gemm(1.0, dense, 0.0, product)
        return product
    }

    @Benchmark
    fun preparedGemv(): DoubleArray {
        prepared.gemv(1.0, x, 0.0, y)
        return y
    }

    /** Reuses every operand handle the selected implementation can prepare; result export remains timed. */
    @Benchmark
    fun preparedSparseProduct(): SparseMatrix = prepared.sparseProduct(preparedSquare)

    @Benchmark
    fun gemv(): DoubleArray {
        arm.gemv(1.0, a, x, 0.0, y)
        return y
    }

    /** Includes operand preparation and representation conversion as a separately labelled one-shot row. */
    @Benchmark
    fun sparseProduct(): SparseMatrix = arm.sparseProduct(a, square)

    @Benchmark
    fun trsv(): DoubleArray {
        triangularVector.copyInto(scratch)
        preparedTriangle.trsv(triangularVector, scratch, triangleTranspose)
        return scratch
    }

    @Benchmark
    fun trmv(): DoubleArray {
        triangularVector.copyInto(scratch)
        preparedTriangle.trmv(triangularVector, scratch, triangleTranspose)
        return scratch
    }

    @Benchmark
    fun trmm(): DenseMatrix {
        triangularDense.data.copyInto(triangularProduct.data)
        preparedTriangle.trmm(triangularDense, triangularProduct, triangleTranspose)
        return triangularProduct
    }

    @Benchmark
    fun trsm(): DenseMatrix {
        triangularDense.data.copyInto(triangularSolve.data)
        preparedTriangle.trsm(triangularDense, triangularSolve, triangleTranspose)
        return triangularSolve
    }

    @Benchmark
    fun trmmRight(): DenseMatrix {
        triangularDenseRight.data.copyInto(triangularProductRight.data)
        preparedTriangle.trmm(triangularDenseRight, triangularProductRight, triangleTranspose, right = true)
        return triangularProductRight
    }

}

/** Triangular storage variants kept outside the immutable contributor profile v1. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseTriangularVariantBenchmark {
    @Param("64", "257") var n: Int = 64
    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND) var sparseArm: String = BUILTIN_BACKEND
    @Param("0.01") var density: Double = 0.01
    @Param("regular") var productShape: String = "regular"
    @Param("upper-nontrans-nonunit") var triangleVariant: String = "upper-nontrans-nonunit"

    private lateinit var delegate: SparseProductHostBenchmark

    @Setup
    fun setup() {
        delegate = SparseProductHostBenchmark().also {
            it.n = n
            it.sparseArm = sparseArm
            it.density = density
            it.productShape = productShape
            it.triangleVariant = triangleVariant
            it.setup()
        }
    }

    @TearDown
    fun tearDown() {
        delegate.tearDown()
    }

    @Benchmark fun trsv(): DoubleArray = delegate.trsv()

    @Benchmark fun trmv(): DoubleArray = delegate.trmv()

    @Benchmark fun trmm(): DenseMatrix = delegate.trmm()

    @Benchmark fun trsm(): DenseMatrix = delegate.trsm()

    @Benchmark fun trmmRight(): DenseMatrix = delegate.trmmRight()
}

/** Right-side sparse triangular solve, separated from the immutable contributor profile v1. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseRightTriangularBenchmark {
    @Param("64", "257") var n: Int = 64
    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND) var sparseArm: String = BUILTIN_BACKEND
    @Param("upper-nontrans-nonunit") var triangleVariant: String = "upper-nontrans-nonunit"

    private lateinit var triangle: SparseMatrix
    private lateinit var input: DenseMatrix
    private lateinit var output: DenseMatrix
    private lateinit var resources: BenchmarkResources
    private lateinit var preparedTriangle: PreparedSparseBenchmarkArm
    private var lower = false
    private var transpose = false
    private var unitDiag = false

    @Setup
    fun setup() {
        val parts = triangleVariant.split('-')
        check(parts.size == 3) { "unknown triangular variant: $triangleVariant" }
        lower = when (parts[0]) {
            "lower" -> true
            "upper" -> false
            else -> error("unknown triangle: ${parts[0]}")
        }
        transpose = when (parts[1]) {
            "trans" -> true
            "nontrans" -> false
            else -> error("unknown transpose: ${parts[1]}")
        }
        unitDiag = when (parts[2]) {
            "unit" -> true
            "nonunit" -> false
            else -> error("unknown diagonal: ${parts[2]}")
        }
        triangle = bandTriangle(n, lower)
        input = randomMatrix(RIGHT_HAND_SIDES, n, benchRng())
        output = DenseMatrix.zero(RIGHT_HAND_SIDES, n)
        val arm = SparseBenchmarkArm.resolve(sparseArm)
        resources = BenchmarkResources()
        preparedTriangle = resources.acquire {
            arm.prepare(triangle, SparseDescriptor(triangular = true, lower = lower, unitDiag = unitDiag))
        }
        reportAllocatingWorkload(
            "sparse/$sparseArm/trsm-right-composition",
            "oneMKL requires dense transpose staging around its left-side sparse solve",
        )
    }

    @TearDown
    fun tearDown() {
        resources.close()
    }

    @Benchmark
    fun trsmRight(): DenseMatrix {
        input.data.copyInto(output.data)
        preparedTriangle.trsm(input, output, transpose, right = true)
        return output
    }
}

/** Right-hand sides for the level-3 row, enough that a multi-column call is not a single-column one. */
private const val RIGHT_HAND_SIDES = 8
