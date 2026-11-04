package com.eignex.koblas.engine

import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.engine.bestEffortProfile

/** Complete target configurations with measured profiles and documented best-effort preferences. */
public object StandardEngineProfiles {
    /** One prebuilt profile, borrowing documented best-effort preferences when the target is unmeasured. */
    @kotlin.jvm.JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun forTarget(runtime: EngineRuntime, host: HostPlatform, nativeAccess: Boolean = true): EngineProfile {
        val selected = profiles.firstOrNull { it.appliesTo(runtime, host) }
            ?: bestEffortProfile(runtime, host, profiles) ?: portable
        return if (nativeAccess) {
            selected
        } else {
            EngineProfile(
                "${selected.name}-owned",
                selected.rules.filter { it.source == "jvm-simd" },
                runtime,
                selected.target,
                selected.evidence,
            )
        }
    }

    private val portable = EngineProfile("portable", emptyList())
    private val profiles = listOf(
        EngineProfile(
            "jvm-linux-arm64-sve256-v1",
            listOf(Vendor.OpenBlas, Vendor.ArmPl).flatMap { it.jvmProductRules(128) } + jvmOwned(
                owned(setOf(Dot, Asum), minimum = 32),
                owned(setOf(Sum, Nrm2), minimum = 16),
                owned(setOf(Iamax), minimum = 256),
                owned(setOf(Axpy), minimum = 1048576),
            ),
            EngineRuntime.Jvm,
            HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 256),
            "https://lab.eignex.com/jobs/10607; " +
                "Coarse estimates; Unmeasured best effort: ArmPL from OpenBLAS",
        ),
        EngineProfile(
            "jvm-linux-x86_64-avx2-v1",
            listOf(Vendor.OpenBlas, Vendor.OneMkl, Vendor.Aocl).flatMap { it.jvmProductRules(128) } + jvmOwned(
                owned(setOf(Dot, Asum, Sum), minimum = 32),
                owned(setOf(Nrm2), minimum = 16),
                owned(setOf(Iamax), minimum = 1024),
                owned(setOf(Axpy), minimum = 1048576),
            ),
            EngineRuntime.Jvm,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx2),
            "https://lab.eignex.com/jobs/10584; " +
                "Coarse estimates; Unmeasured best effort: AOCL from OpenBLAS",
        ),
        EngineProfile(
            "jvm-linux-x86_64-avx512-v1",
            listOf(Vendor.OneMkl, Vendor.OpenBlas, Vendor.Aocl).flatMap { it.jvmProductRules(64) } + jvmOwned(
                owned(setOf(Dot, Asum), minimum = 8),
                owned(setOf(Nrm2), minimum = 16),
                owned(setOf(Sum), minimum = 32),
                owned(setOf(Iamax), minimum = 1024),
                owned(setOf(Axpy), minimum = 1048576),
            ),
            EngineRuntime.Jvm,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512),
            "https://lab.eignex.com/jobs/10566; " +
                "Coarse estimates; Unmeasured best effort: AOCL from OpenBLAS",
        ),
        EngineProfile(
            "jvm-macos-arm64-v1",
            Vendor.Accelerate.jvmProductRules(64) + jvmOwned(
                owned(setOf(Dot, Sum, Asum, Nrm2, Axpy), minimum = 32),
                owned(setOf(Iamax), minimum = 256),
            ),
            EngineRuntime.Jvm,
            HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, cpuClass = CpuClass.Arm64),
            "BACKENDS.md; " +
                "Coarse estimates; Unmeasured best effort: entire macOS JVM profile",
        ),
        EngineProfile(
            "native-linux-arm64-sve256-v1",
            listOf(
                Vendor.OpenBlas.rule(setOf(Asum), rows = 1024, contiguous = true),
                Vendor.OpenBlas.rule(setOf(Axpy, Rot, Swap), rows = 4194304, contiguous = true),
                Vendor.OpenBlas.rule(setOf(Dot), rows = 4194304, contiguous = false),
                Vendor.OpenBlas.rule(setOf(Dot), rows = 64, contiguous = true),
                Vendor.OpenBlas.rule(
                    setOf(Gemm, Gemmt, Symm, Syr2k, Syrk, Trmm, Trsm),
                    rows = 8,
                    columns = 8,
                    depth = 8,
                    work = 512,
                    borrowed = false,
                ),
                Vendor.OpenBlas.rule(setOf(Gemm), rows = 16, columns = 16, depth = 16, work = 65536, borrowed = true),
                Vendor.OpenBlas.rule(
                    setOf(Gemv, GemvTransposed, Ger),
                    rows = 16,
                    columns = 16,
                    work = 1024,
                    borrowed = false,
                ),
                Vendor.OpenBlas.rule(setOf(Iamax), rows = 256, contiguous = false),
                Vendor.OpenBlas.rule(setOf(Iamax), rows = 96, contiguous = true),
                Vendor.OpenBlas.rule(setOf(Nrm2, Scale), rows = 256, contiguous = true),
                Vendor.OpenBlas.rule(
                    setOf(Symv, Syr, Trmv, TrmvTransposed, Trsv, TrsvTransposed),
                    rows = 32,
                    columns = 32,
                    work = 1024,
                    borrowed = false,
                ),
                Vendor.OpenBlas.rule(setOf(Syr2), rows = 16, columns = 16, work = 256, borrowed = false),
            ),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 256),
            "https://lab.eignex.com/jobs/10628; " +
                "Coarse estimates informed by retained measurements; favor host on Native",
        ),
        EngineProfile(
            "native-linux-x86_64-avx2-v1",
            Vendor.OpenBlas.nativeX86Rules(reduction = 128, norm = 256, level2 = 64) +
                Vendor.OneMkl.nativeX86Rules(reduction = 256, norm = 512, level2 = 32),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx2),
            "https://lab.eignex.com/jobs/10583; " +
                "Coarse estimates informed by retained measurements; favor host on Native",
        ),
        EngineProfile(
            "native-linux-x86_64-avx512-v1",
            Vendor.OneMkl.nativeX86Rules(reduction = 256, norm = 512, level2 = 32) +
                Vendor.OpenBlas.nativeX86Rules(reduction = 128, norm = 256, level2 = 64),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512),
            "https://lab.eignex.com/jobs/10565; " +
                "Coarse estimates informed by retained measurements; favor host on Native",
        ),
        EngineProfile(
            "native-macos-arm64-v1",
            listOf(
                Vendor.Accelerate.rule(setOf(Asum, Iamax), rows = 64, contiguous = true),
                Vendor.Accelerate.rule(setOf(Axpy), rows = 4194304, contiguous = true),
                Vendor.Accelerate.rule(setOf(Dot, Nrm2), rows = 96, contiguous = true),
                Vendor.Accelerate.rule(
                    setOf(Gemm, Symm, Syrk, Trmm, Trsm),
                    rows = 8,
                    columns = 8,
                    depth = 8,
                    work = 512,
                    borrowed = false,
                ),
                Vendor.Accelerate.rule(setOf(Gemm), rows = 16, columns = 16, depth = 16, work = 65536, borrowed = true),
                Vendor.Accelerate.rule(
                    setOf(Gemv, GemvTransposed),
                    rows = 16,
                    columns = 16,
                    work = 4096,
                    borrowed = false,
                ),
                Vendor.Accelerate.rule(setOf(Ger), rows = 16, columns = 16, work = 1024, borrowed = false),
                Vendor.Accelerate.rule(setOf(Swap), rows = 4194304, contiguous = true),
                Vendor.Accelerate.rule(setOf(Symv), rows = 256, columns = 256, work = 65536, borrowed = false),
                Vendor.Accelerate.rule(
                    setOf(Syr, Syr2, TrmvTransposed, Trsv, TrsvTransposed),
                    rows = 32,
                    columns = 32,
                    work = 1024,
                    borrowed = false,
                ),
                Vendor.Accelerate.rule(
                    setOf(Syr2k),
                    rows = 16,
                    columns = 16,
                    depth = 16,
                    work = 4096,
                    borrowed = false,
                ),
                Vendor.Accelerate.rule(setOf(Trmv), rows = 64, columns = 64, work = 4096, borrowed = false),
            ),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, cpuClass = CpuClass.Arm64),
            "https://lab.eignex.com/jobs/10630; " +
                "Coarse estimates informed by retained measurements; favor host on Native",
        ),
    )

    private fun Vendor.jvmProductRules(minimum: Int): List<EngineRule> = listOf(
        rule(setOf(Gemm), rows = minimum, columns = minimum, depth = minimum, work = 262144),
        rule(setOf(Gemm), rows = 16, columns = 16, depth = 256, work = 65536, borrowed = false),
        rule(
            setOf(Symm, Syrk, Syr2k, Trmm, Trsm),
            rows = minimum,
            columns = minimum,
            depth = minimum,
            work = 1048576,
            borrowed = false,
        ),
    )

    private fun Vendor.nativeX86Rules(reduction: Int, norm: Int, level2: Int): List<EngineRule> = listOf(
        rule(setOf(Dot, Asum), rows = reduction, contiguous = true),
        rule(setOf(Nrm2, Iamax), rows = norm, contiguous = true),
        rule(setOf(Axpy, Scale, Swap, Rot), rows = 4096, contiguous = true),
        rule(setOf(Gemm, Symm, Syr2k, Trmm, Trsm), rows = 8, columns = 8, depth = 8, work = 512, borrowed = false),
        rule(setOf(Gemm), rows = 16, columns = 16, depth = 16, work = 65536, borrowed = true),
        rule(setOf(Gemmt, Syrk), rows = 16, columns = 16, depth = 16, work = 4096, borrowed = false),
        rule(
            setOf(Gemv, GemvTransposed, Ger, Symv, Syr, Syr2, Trmv, TrmvTransposed, Trsv, TrsvTransposed),
            rows = level2,
            columns = level2,
            work = level2.toLong() * level2,
            borrowed = false,
        ),
    )

    private fun Vendor.rule(
        implementations: Set<EngineImplementation>,
        rows: Int = 0,
        columns: Int = 0,
        depth: Int = 0,
        work: Long = 0,
        contiguous: Boolean? = null,
        borrowed: Boolean? = null,
    ): EngineRule = EngineRule(
        vendorName.lowercase(),
        implementations,
        rows..Int.MAX_VALUE,
        columns..Int.MAX_VALUE,
        depth..Int.MAX_VALUE,
        work,
        contiguous,
        borrowed,
    )

    private fun owned(
        implementations: Set<EngineImplementation>,
        minimum: Int = 0,
        contiguous: Boolean? = true,
    ): EngineRule = EngineRule("jvm-simd", implementations, rows = minimum..Int.MAX_VALUE, contiguous = contiguous)

    /** Measured Level 1 rules, then the indexed dot and scheduled families every JVM profile keeps owned. */
    private fun jvmOwned(vararg level1: EngineRule): List<EngineRule> = level1.asList() +
        owned(setOf(IndexedDot), minimum = 16, contiguous = null) +
        owned(setOf(Panels, Products, Triangular), contiguous = null)
}
