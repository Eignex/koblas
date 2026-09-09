package com.eignex.koblas.bench

import com.eignex.koblas.KoblasContext
import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.DenseMatrix

internal const val BUILTIN_BACKEND = "built-in"
internal const val SCALAR_BACKEND = "scalar"
internal const val C_BACKEND = "c"
internal const val SIMD_BACKEND = "simd"
internal const val OPENBLAS_BACKEND = "openblas"
internal const val ONEMKL_BACKEND = "onemkl"

/** A benchmark-only dense implementation, selected without production backend discovery. */
internal interface DenseComparator {
    val identity: String
    val threading: String

    fun dot(x: DoubleArray, y: DoubleArray): Double
    fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray)
    fun scale(alpha: Double, x: DoubleArray)
    fun nrm2(x: DoubleArray): Double
    fun asum(x: DoubleArray): Double
    fun swap(x: DoubleArray, y: DoubleArray)
    fun rotm(x: DoubleArray, y: DoubleArray, transformation: ModifiedGivens)
    fun rot(x: DoubleArray, y: DoubleArray, c: Double, s: Double)

    fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean)
    fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean)
    fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix)
    fun syr(alpha: Double, x: DoubleArray, a: DenseMatrix, lower: Boolean)
    fun syr2(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, lower: Boolean)
    fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean)
    fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean)

    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    )

    fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean)
    fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    )

    fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    )

    fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    )

    fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    )
}

internal expect fun explicitBuiltInContext(): KoblasContext
internal expect fun openBlasComparator(): DenseComparator?
internal expect fun oneMklDenseComparator(): DenseComparator?

internal class DenseBenchmarkArm private constructor(
    val context: KoblasContext?,
    val external: DenseComparator?,
) {
    val identity: String get() = external?.identity ?: "built-in/${context!!.blas.name}/${context.kernels.name}"
    val threading: String get() = external?.threading ?: "single calling thread"

    companion object {
        fun resolve(name: String): DenseBenchmarkArm {
            val arm = when (name) {
                BUILTIN_BACKEND -> DenseBenchmarkArm(explicitBuiltInContext(), null)
                SCALAR_BACKEND, C_BACKEND, SIMD_BACKEND -> DenseBenchmarkArm(kernelEngine(name), null)
                OPENBLAS_BACKEND -> DenseBenchmarkArm(
                    null,
                    checkNotNull(openBlasComparator()) { "the benchmark-only single-threaded OpenBLAS comparator is unavailable" },
                )
                ONEMKL_BACKEND -> DenseBenchmarkArm(
                    null,
                    checkNotNull(oneMklDenseComparator()) { "the benchmark-only single-threaded oneMKL comparator is unavailable" },
                )
                else -> error("unknown dense benchmark arm: $name")
            }
            check(
                (name in setOf(BUILTIN_BACKEND, SCALAR_BACKEND, C_BACKEND, SIMD_BACKEND) &&
                    arm.context != null && arm.external == null) ||
                    (name in setOf(OPENBLAS_BACKEND, ONEMKL_BACKEND) && arm.context == null && arm.external != null),
            ) { "benchmark arm '$name' did not resolve to its named implementation" }
            println("resolved: arm=$name dense=${arm.identity} threading=${arm.threading}")
            return arm
        }
    }
}
