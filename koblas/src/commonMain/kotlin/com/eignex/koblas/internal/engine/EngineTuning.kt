package com.eignex.koblas.internal.engine

import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.EngineImplementation.*

/** Ordered internal cutoffs and their retained measurement provenance. */
internal data class EngineTuning(
    val name: String,
    val rules: List<EngineRule>,
    val evidence: String = "explicit configuration",
)

/** Select fixed tuning by runtime and instruction class; other hosts borrow compatible estimates. */
internal fun standardTuning(runtime: EngineRuntime, host: HostPlatform, nativeAccess: Boolean = true): EngineTuning {
    if (host.architecture == Architecture.Other) return EngineTuning("portable", emptyList())
    val x86 = host.architecture == Architecture.X86_64
    val wide = host.cpuClass == CpuClass.Avx512
    val mac = host.operatingSystem == OperatingSystem.MacOs
    val owned = when {
        runtime == EngineRuntime.Native -> null
        x86 -> if (wide) jvmAvx512 else jvmAvx2
        host.cpuClass == CpuClass.Sve -> jvmArm
        else -> jvmMac
    }
    val vendor = when {
        mac -> if (runtime == EngineRuntime.Jvm) jvmMac else nativeMac
        host.operatingSystem == OperatingSystem.Other -> null
        runtime == EngineRuntime.Jvm -> if (x86) owned else jvmArm
        x86 -> if (wide) nativeAvx512 else nativeAvx2
        else -> nativeArm
    }
    val donors = listOfNotNull(vendor, owned).distinct()
    if (donors.isEmpty()) return EngineTuning("portable", emptyList())
    val measured = when {
        mac -> !x86 && host.cpuClass == CpuClass.Arm64
        host.operatingSystem != OperatingSystem.Linux -> false
        x86 -> host.cpuClass == CpuClass.Avx2 || wide
        else -> host.cpuClass == CpuClass.Sve && host.sveVectorBits == 256
    }
    val name = if (measured) {
        requireNotNull(vendor).name
    } else {
        "best-effort-$runtime-${host.operatingSystem}-${host.architecture}-${donors.joinToString("+") { it.name }}"
    }
    val rules = vendor?.rules.orEmpty().filter { it.source != "jvm-simd" && nativeAccess } +
        owned?.rules.orEmpty().filter { it.source == "jvm-simd" }
    val evidence = if (measured) {
        requireNotNull(vendor).evidence
    } else {
        "Unmeasured best effort; " + donors.joinToString { "preferences from ${it.name}" }
    }
    return EngineTuning(if (nativeAccess) name else "$name-owned", rules, evidence)
}

private val jvmArm = EngineTuning(
    "jvm-linux-arm64-sve256-v1",
    listOf(Vendor.OpenBlas, Vendor.ArmPl).flatMap { it.jvmProductRules(128) } + jvmOwned(
        owned(setOf(Dot, Asum), minimum = 32),
        owned(setOf(Sum, Nrm2), minimum = 16),
        owned(setOf(Iamax), minimum = 256),
        owned(setOf(Axpy), minimum = 1048576),
    ),
    "https://lab.eignex.com/jobs/10607; " +
        "Coarse estimates; Unmeasured best effort: ArmPL from OpenBLAS",
)

private val jvmAvx2 = EngineTuning(
    "jvm-linux-x86_64-avx2-v1",
    listOf(Vendor.OpenBlas, Vendor.OneMkl, Vendor.Aocl).flatMap { it.jvmProductRules(128) } + jvmOwned(
        owned(setOf(Dot, Asum, Sum), minimum = 32),
        owned(setOf(Nrm2), minimum = 16),
        owned(setOf(Iamax), minimum = 1024),
        owned(setOf(Axpy), minimum = 1048576),
    ),
    "https://lab.eignex.com/jobs/10584; " +
        "Coarse estimates; Unmeasured best effort: AOCL from OpenBLAS",
)

private val jvmAvx512 = EngineTuning(
    "jvm-linux-x86_64-avx512-v1",
    listOf(Vendor.OneMkl, Vendor.OpenBlas, Vendor.Aocl).flatMap { it.jvmProductRules(64) } + jvmOwned(
        owned(setOf(Dot, Asum), minimum = 8),
        owned(setOf(Nrm2), minimum = 16),
        owned(setOf(Sum), minimum = 32),
        owned(setOf(Iamax), minimum = 1024),
        owned(setOf(Axpy), minimum = 1048576),
    ),
    "https://lab.eignex.com/jobs/10566; " +
        "Coarse estimates; Unmeasured best effort: AOCL from OpenBLAS",
)

private val jvmMac = EngineTuning(
    "jvm-macos-arm64-v1",
    Vendor.Accelerate.jvmProductRules(64) + jvmOwned(
        owned(setOf(Dot, Sum, Asum, Nrm2, Axpy), minimum = 32),
        owned(setOf(Iamax), minimum = 256),
    ),
    "BACKENDS.md; " +
        "Coarse estimates; Unmeasured best effort: entire macOS JVM profile",
)

private val nativeArm = EngineTuning(
    "native-linux-arm64-sve256-v1",
    listOf(
        Vendor.OpenBlas.rule(setOf(Asum), rows = 1024, contiguous = true),
        Vendor.OpenBlas.rule(setOf(Axpy, Rot, Swap), rows = 4194304, contiguous = true),
        Vendor.OpenBlas.rule(setOf(Dot), rows = 4194304, contiguous = false),
        Vendor.OpenBlas.rule(setOf(Dot), rows = 64, contiguous = true),
        Vendor.OpenBlas.matrix(setOf(Gemm, Gemmt, Symm, Syr2k, Syrk, Trmm, Trsm), 8, 512),
        Vendor.OpenBlas.matrix(setOf(Gemm), 16, 65536, borrowed = true),
        Vendor.OpenBlas.matrix(setOf(Gemv, GemvTransposed, Ger), 16, 1024, depth = 0),
        Vendor.OpenBlas.rule(setOf(Iamax), rows = 256, contiguous = false),
        Vendor.OpenBlas.rule(setOf(Iamax), rows = 96, contiguous = true),
        Vendor.OpenBlas.rule(setOf(Nrm2, Scale), rows = 256, contiguous = true),
        Vendor.OpenBlas.matrix(setOf(Symv, Syr, Trmv, TrmvTransposed, Trsv, TrsvTransposed), 32, 1024, depth = 0),
        Vendor.OpenBlas.matrix(setOf(Syr2), 16, 256, depth = 0),
    ),
    "https://lab.eignex.com/jobs/10628; coarse estimates informed by retained measurements; favor host on Native",
)

private val nativeAvx2 = EngineTuning(
    "native-linux-x86_64-avx2-v1",
    Vendor.OpenBlas.nativeX86Rules(reduction = 128, norm = 256, level2 = 64) +
        Vendor.OneMkl.nativeX86Rules(reduction = 256, norm = 512, level2 = 32),
    "https://lab.eignex.com/jobs/10583; " +
        "Coarse estimates informed by retained measurements; favor host on Native",
)

private val nativeAvx512 = EngineTuning(
    "native-linux-x86_64-avx512-v1",
    Vendor.OneMkl.nativeX86Rules(reduction = 256, norm = 512, level2 = 32) +
        Vendor.OpenBlas.nativeX86Rules(reduction = 128, norm = 256, level2 = 64),
    "https://lab.eignex.com/jobs/10565; " +
        "Coarse estimates informed by retained measurements; favor host on Native",
)

private val nativeMac = EngineTuning(
    "native-macos-arm64-v1",
    listOf(
        Vendor.Accelerate.rule(setOf(Asum, Iamax), rows = 64, contiguous = true),
        Vendor.Accelerate.rule(setOf(Axpy), rows = 4194304, contiguous = true),
        Vendor.Accelerate.rule(setOf(Dot, Nrm2), rows = 96, contiguous = true),
        Vendor.Accelerate.matrix(setOf(Gemm, Symm, Syrk, Trmm, Trsm), 8, 512),
        Vendor.Accelerate.matrix(setOf(Gemm), 16, 65536, borrowed = true),
        Vendor.Accelerate.matrix(setOf(Gemv, GemvTransposed), 16, 4096, depth = 0),
        Vendor.Accelerate.matrix(setOf(Ger), 16, 1024, depth = 0),
        Vendor.Accelerate.rule(setOf(Swap), rows = 4194304, contiguous = true),
        Vendor.Accelerate.matrix(setOf(Symv), 256, 65536, depth = 0),
        Vendor.Accelerate.matrix(setOf(Syr, Syr2, TrmvTransposed, Trsv, TrsvTransposed), 32, 1024, depth = 0),
        Vendor.Accelerate.matrix(setOf(Syr2k), 16, 4096),
        Vendor.Accelerate.matrix(setOf(Trmv), 64, 4096, depth = 0),
    ),
    "https://lab.eignex.com/jobs/10630; coarse estimates informed by retained measurements; favor host on Native",
)

private fun Vendor.jvmProductRules(minimum: Int): List<EngineRule> = listOf(
    matrix(setOf(Gemm), minimum, 262144, borrowed = null),
    matrix(setOf(Gemm), 16, 65536, depth = 256),
    matrix(setOf(Symm, Syrk, Syr2k, Trmm, Trsm), minimum, 1048576),
)

private fun Vendor.nativeX86Rules(reduction: Int, norm: Int, level2: Int): List<EngineRule> = listOf(
    rule(setOf(Dot, Asum), rows = reduction, contiguous = true),
    rule(setOf(Nrm2, Iamax), rows = norm, contiguous = true),
    rule(setOf(Axpy, Scale, Swap, Rot), rows = 4096, contiguous = true),
    matrix(setOf(Gemm, Symm, Syr2k, Trmm, Trsm), 8, 512),
    matrix(setOf(Gemm), 16, 65536, borrowed = true),
    matrix(setOf(Gemmt, Syrk), 16, 4096),
    matrix(
        setOf(Gemv, GemvTransposed, Ger, Symv, Syr, Syr2, Trmv, TrmvTransposed, Trsv, TrsvTransposed),
        level2,
        level2.toLong() * level2,
        depth = 0,
    ),
)

private fun Vendor.rule(operations: Set<EngineImplementation>, rows: Int, contiguous: Boolean): EngineRule =
    EngineRule(vendorName.lowercase(), operations, rows = rows, contiguous = contiguous)

/** Matrix cutoffs share a minimum row and column extent; vector products have no depth cutoff. */
private fun Vendor.matrix(
    operations: Set<EngineImplementation>,
    minimum: Int,
    work: Long,
    depth: Int = minimum,
    borrowed: Boolean? = false,
): EngineRule = EngineRule(vendorName.lowercase(), operations, minimum, minimum, depth, work, borrowed = borrowed)

private fun owned(
    implementations: Set<EngineImplementation>,
    minimum: Int = 0,
    contiguous: Boolean? = true,
): EngineRule = EngineRule("jvm-simd", implementations, rows = minimum, contiguous = contiguous)

/** Measured Level 1 rules, then the indexed dot and scheduled families every JVM profile keeps owned. */
private fun jvmOwned(vararg level1: EngineRule): List<EngineRule> = level1.asList() +
    owned(setOf(IndexedDot), minimum = 16, contiguous = null) +
    owned(setOf(Panels, Products, Triangular), contiguous = null)
