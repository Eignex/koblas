package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
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

    private lateinit var a: F64DenseMatrix
    private lateinit var b: F64DenseMatrix
    private lateinit var c: F64DenseMatrix

    /** Symmetric and triangular operands, plus a destination the triangular routines overwrite. */
    private lateinit var sym: F64DenseMatrix
    private lateinit var triangular: F64DenseMatrix
    private lateinit var rhs: F64DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        a = randomMatrix(n, n, rng)
        b = randomMatrix(n, n, rng)
        c = F64DenseMatrix.zero(n, n)
        sym = lowerSymmetricMatrix(n, rng)
        triangular = dominantMatrix(n, rng)
        rhs = F64DenseMatrix.zero(n, n)
    }

    @Benchmark
    fun gemm(): F64DenseMatrix = a.matMul(b)

    @Benchmark
    fun syrkFull(): F64DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c)
        return c
    }

    @Benchmark
    fun syrkLower(): F64DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c, uplo = Uplo.LOWER)
        return c
    }

    /** Rank-2k, two dots per entry over operands transposed into scratch, and no workspace to lend it. */
    @Benchmark
    fun syr2k(): F64DenseMatrix {
        koblas.syr2k(1.0, a, b, transpose = false, beta = 0.0, c = c, uplo = Uplo.LOWER)
        return c
    }

    @Benchmark
    fun symm(): F64DenseMatrix {
        koblas.symm(1.0, sym, b, 0.0, c)
        return c
    }

    @Benchmark
    fun symmRight(): F64DenseMatrix {
        koblas.symm(1.0, sym, b, 0.0, c, right = true)
        return c
    }

    /** Triangular solve with many right-hand sides, the blocked path a host library wins on. */
    @Benchmark
    fun trsm(): F64DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trsm(triangular, rhs, lower = true)
        return rhs
    }

    @Benchmark
    fun trmm(): F64DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trmm(triangular, rhs, lower = true)
        return rhs
    }

    /** The right-side form, which the portable path serves one row at a time rather than one column. */
    @Benchmark
    fun trsmRight(): F64DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trsm(triangular, rhs, lower = true, right = true)
        return rhs
    }

    @Benchmark
    fun trmmRight(): F64DenseMatrix {
        b.data.copyInto(rhs.data)
        koblas.trmm(triangular, rhs, lower = true, right = true)
        return rhs
    }
}
