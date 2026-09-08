package com.eignex.koblas.bench

import com.eignex.koblas.F64Context
import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.core.F64DenseMatrix

internal const val BUILTIN_BACKEND = "built-in"
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
    fun rotm(x: DoubleArray, y: DoubleArray, transformation: F64ModifiedGivens)
    fun rot(x: DoubleArray, y: DoubleArray, c: Double, s: Double)

    fun gemv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean)
    fun symv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean)
    fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix)
    fun syr(alpha: Double, x: DoubleArray, a: F64DenseMatrix, lower: Boolean)
    fun syr2(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix, lower: Boolean)
    fun trsv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean)
    fun trmv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean)

    fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    )

    fun syrk(alpha: Double, a: F64DenseMatrix, transpose: Boolean, beta: Double, c: F64DenseMatrix, lower: Boolean)
    fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
    )

    fun symm(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        right: Boolean,
    )

    fun trsm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    )

    fun trmm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    )
}

internal expect fun explicitBuiltInContext(): F64Context
internal expect fun openBlasComparator(): DenseComparator?
internal expect fun oneMklDenseComparator(): DenseComparator?

internal class DenseBenchmarkArm private constructor(
    val context: F64Context?,
    val external: DenseComparator?,
) {
    val identity: String get() = external?.identity ?: "built-in/${context!!.blas.name}/${context.kernels.name}"
    val threading: String get() = external?.threading ?: "single calling thread"

    companion object {
        fun resolve(name: String): DenseBenchmarkArm {
            val arm = when (name) {
                BUILTIN_BACKEND -> DenseBenchmarkArm(explicitBuiltInContext(), null)
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
                (name == BUILTIN_BACKEND && arm.context != null && arm.external == null) ||
                    (name in setOf(OPENBLAS_BACKEND, ONEMKL_BACKEND) && arm.context == null && arm.external != null),
            ) { "benchmark arm '$name' did not resolve to its named implementation" }
            println("resolved: arm=$name dense=${arm.identity} threading=${arm.threading}")
            return arm
        }
    }
}
