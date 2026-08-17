package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.matMul
import com.eignex.koblas.koblas
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level3Benchmark {
    @Param("4", "16", "32", "64", "128", "256")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var c: DenseMatrix

    /** Symmetric and triangular operands, plus a destination the triangular routines overwrite. */
    private lateinit var sym: DenseMatrix
    private lateinit var triangular: DenseMatrix
    private lateinit var rhs: DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        a = randomMatrix(n, n, rng)
        b = randomMatrix(n, n, rng)
        c = DenseMatrix.zero(n, n)
        sym = lowerSymmetricMatrix(n, rng)
        triangular = dominantMatrix(n, rng)
        rhs = DenseMatrix.zero(n, n)
    }

    @Benchmark
    fun gemm(): DenseMatrix = a.matMul(b)

    @Benchmark
    fun syrkFull(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c)
        return c
    }

    @Benchmark
    fun syrkLower(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c, uplo = Uplo.LOWER)
        return c
    }

    /** Rank-2k, whose inner loop reads both operands through the unchecked accessor. */
    @Benchmark
    fun syr2k(): DenseMatrix {
        koblas.syr2k(1.0, a, b, transpose = false, beta = 0.0, c = c, uplo = Uplo.LOWER)
        return c
    }

    @Benchmark
    fun symm(): DenseMatrix {
        koblas.symm(1.0, sym, b, 0.0, c)
        return c
    }

    @Benchmark
    fun symmRight(): DenseMatrix {
        koblas.symm(1.0, sym, b, 0.0, c, right = true)
        return c
    }

    /** Triangular solve with many right-hand sides, the blocked path a host library wins on. */
    @Benchmark
    fun trsm(): DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trsm(triangular, rhs, lower = true)
        return rhs
    }

    @Benchmark
    fun trmm(): DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trmm(triangular, rhs, lower = true)
        return rhs
    }
}
