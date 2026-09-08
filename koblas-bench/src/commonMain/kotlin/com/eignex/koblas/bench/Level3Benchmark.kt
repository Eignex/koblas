package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level3Benchmark {
    // Through 256 the operands stay in L2, where the cache tiles cannot show; 512 and 1024 are where
    // blocking either works or does not.
    @Param("4", "16", "31", "32", "33", "63", "64", "65", "127", "128", "129", "255", "256", "257", "512", "1024")
    var n: Int = 0

    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var denseArm: String = BUILTIN_BACKEND

    private lateinit var arm: DenseBenchmarkArm

    private lateinit var a: DenseMatrix
    private lateinit var transposedA: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var c: DenseMatrix

    private lateinit var squareA: DenseMatrix
    private lateinit var squareB: DenseMatrix

    private lateinit var sym: DenseMatrix
    private lateinit var triangular: DenseMatrix
    private lateinit var rhs: DenseMatrix

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val rng = benchRng()
        // Deliberately rectangular and off the tile width, including when n itself is a tile boundary.
        a = randomMatrix(n + 1, n - 1, rng)
        transposedA = randomMatrix(n - 1, n + 1, rng)
        b = randomMatrix(n - 1, n + 3, rng)
        c = DenseMatrix.zero(n + 1, n + 3)
        squareA = randomMatrix(n, n, rng)
        squareB = randomMatrix(n, n, rng)
        sym = lowerSymmetricMatrix(n, rng)
        triangular = dominantMatrix(n, rng)
        rhs = DenseMatrix.zero(n, n)
        reportAllocatingWorkload("level3/$denseArm/gemm", "built-in packing workspace or fresh result construction")
    }

    @Benchmark
    fun gemm(): DenseMatrix {
        arm.external?.gemm(1.0, a, false, b, false, 0.0, c) ?: arm.context!!.gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    /** The transposed-left panel update, which the plain [gemm] above never reaches. */
    @Benchmark
    fun gemmTransposedA(): DenseMatrix {
        arm.external?.gemm(1.0, transposedA, true, b, false, 0.0, c) ?: arm.context!!.gemm(1.0, transposedA, true, b, false, 0.0, c)
        return c
    }

    @Benchmark
    fun symm(): DenseMatrix {
        arm.external?.symm(1.0, sym, squareB, 0.0, rhs, true, false) ?: arm.context!!.symm(1.0, sym, squareB, 0.0, rhs)
        return rhs
    }

    @Benchmark
    fun symmRight(): DenseMatrix {
        arm.external?.symm(1.0, sym, squareB, 0.0, rhs, true, true) ?: arm.context!!.symm(1.0, sym, squareB, 0.0, rhs, right = true)
        return rhs
    }

    @Benchmark
    fun trmm(): DenseMatrix {
        squareB.data.copyInto(rhs.data)
        arm.external?.trmm(triangular, rhs, true, false, false, false, 1.0) ?: arm.context!!.trmm(triangular, rhs, lower = true)
        return rhs
    }

    @Benchmark
    fun trmmRight(): DenseMatrix {
        squareB.data.copyInto(rhs.data)
        arm.external?.trmm(triangular, rhs, true, false, false, true, 1.0) ?: arm.context!!.trmm(triangular, rhs, lower = true, right = true)
        return rhs
    }
}
