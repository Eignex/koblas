package com.eignex.koblas.internal.engine

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.Vendor

/**
 * Fixed crossovers for one selected library and the owned JVM kernels.
 *
 * Retained measurements: lab.eignex.com/jobs/10584, 10566, 10607, 10583, 10565, 10628 and 10630.
 * AOCL, ArmPL and unmeasured platforms use the compatible library or instruction-class estimates.
 */
internal class EngineTuning(
    private val runtime: EngineRuntime,
    private val platform: HostPlatform,
    private val vendor: Vendor?,
) {
    private val x86 = platform.architecture == Architecture.X86_64
    private val wide = platform.cpuClass == CpuClass.Avx512
    private val arm = !x86 && platform.cpuClass != CpuClass.Sve

    fun simdMinimum(operation: EngineImplementation): Int = when (operation) {
        Dot, Asum -> if (wide) 8 else 32
        Sum -> if (!x86 && !arm) 16 else 32
        Nrm2 -> if (arm) 32 else 16
        Iamax -> if (x86) 1024 else 256
        Axpy -> if (arm) 32 else 1048576
        IndexedDot -> 16
        else -> 0
    }

    fun hostMinimum(operation: EngineImplementation, contiguous: Boolean): Int {
        if (runtime == EngineRuntime.Jvm) return -1
        if (vendor == Vendor.Accelerate) {
            return if (!contiguous) {
                -1
            } else {
                when (operation) {
                    Asum, Iamax -> 64
                    Dot, Nrm2 -> 96
                    Axpy, Swap -> 4194304
                    else -> -1
                }
            }
        }
        if (!x86) {
            return when (operation) {
                Dot -> if (contiguous) 64 else 4194304
                Iamax -> if (contiguous) 96 else 256
                Asum -> if (contiguous) 1024 else -1
                Nrm2, Scale -> if (contiguous) 256 else -1
                Axpy, Rot, Swap -> if (contiguous) 4194304 else -1
                else -> -1
            }
        }
        if (!contiguous) return -1
        return when (operation) {
            Dot, Asum -> if (vendor == Vendor.OneMkl) 256 else 128
            Nrm2, Iamax -> if (vendor == Vendor.OneMkl) 512 else 256
            Axpy, Scale, Swap, Rot -> 4096
            else -> -1
        }
    }

    fun hostEligible(
        operation: EngineImplementation,
        rows: Int,
        columns: Int,
        depth: Int,
        borrowed: Boolean,
    ): Boolean {
        fun matrix(minimum: Int, work: Long, shared: Int = minimum): Boolean =
            rows >= minimum && columns >= minimum && depth >= shared && hasWork(rows, columns, depth, work)
        if (runtime == EngineRuntime.Jvm) {
            val minimum = if (wide || platform.operatingSystem == OperatingSystem.MacOs) 64 else 128
            return when (operation) {
                Gemm -> matrix(minimum, 262144) || (!borrowed && matrix(16, 65536, shared = 256))
                Symm, Syrk, Syr2k, Trmm, Trsm -> !borrowed && matrix(minimum, 1048576)
                else -> false
            }
        }
        if (borrowed) return operation == Gemm && matrix(16, 65536)
        if (vendor == Vendor.Accelerate) {
            return when (operation) {
                Gemm, Symm, Syrk, Trmm, Trsm -> matrix(8, 512)
                Syr2k -> matrix(16, 4096)
                Gemv, GemvTransposed -> matrix(16, 4096, shared = 0)
                Ger -> matrix(16, 1024, shared = 0)
                Symv -> matrix(256, 65536, shared = 0)
                Syr, Syr2, TrmvTransposed, Trsv, TrsvTransposed -> matrix(32, 1024, shared = 0)
                Trmv -> matrix(64, 4096, shared = 0)
                else -> false
            }
        }
        if (!x86) {
            return when (operation) {
                Gemm, Gemmt, Symm, Syr2k, Syrk, Trmm, Trsm -> matrix(8, 512)
                Gemv, GemvTransposed, Ger -> matrix(16, 1024, shared = 0)
                Symv, Syr, Trmv, TrmvTransposed, Trsv, TrsvTransposed -> matrix(32, 1024, shared = 0)
                Syr2 -> matrix(16, 256, shared = 0)
                else -> false
            }
        }
        val minimum = if (vendor == Vendor.OneMkl) 32 else 64
        return when (operation) {
            Gemm, Symm, Syr2k, Trmm, Trsm -> matrix(8, 512)

            Gemmt, Syrk -> matrix(16, 4096)

            Gemv, GemvTransposed, Ger, Symv, Syr, Syr2, Trmv, TrmvTransposed, Trsv, TrsvTransposed ->
                matrix(minimum, minimum.toLong() * minimum, shared = 0)

            else -> false
        }
    }
}

/** Compare work without multiplying three potentially maximum extents. */
private fun hasWork(rows: Int, columns: Int, depth: Int, minimum: Long): Boolean = depth > 0 &&
    rows.toLong() * columns >= minimum / depth + if (minimum % depth == 0L) 0 else 1

internal fun vendorOrder(runtime: EngineRuntime, platform: HostPlatform): List<Vendor> = when {
    platform.architecture == Architecture.Other -> emptyList()

    platform.operatingSystem == OperatingSystem.MacOs -> listOf(Vendor.Accelerate)

    platform.operatingSystem == OperatingSystem.Other -> emptyList()

    platform.architecture == Architecture.Arm64 -> listOfNotNull(
        Vendor.OpenBlas,
        Vendor.ArmPl.takeIf { runtime == EngineRuntime.Jvm },
    )

    else -> (
        if (platform.cpuClass == CpuClass.Avx512) {
            listOf(Vendor.OneMkl, Vendor.OpenBlas)
        } else {
            listOf(
                Vendor.OpenBlas,
                Vendor.OneMkl,
            )
        }
        ) + listOfNotNull(Vendor.Aocl.takeIf { runtime == EngineRuntime.Jvm })
}
