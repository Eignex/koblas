package com.eignex.koblas

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.sparse.SparseTuning
import com.eignex.koblas.vendor.*

/** Standard configurations; host acceleration applies only to the declared measurement targets. */
public object StandardEngineProfiles {
    private val intelLinux = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Intel)
    private val levelThree = setOf(Gemm, Gemmt, Symm, Syrk, Syr2k, Trmm, Trsm)
    private val halfWork = setOf(Syr, Trmv, TrmvTransposed, Trsv, TrsvTransposed, Gemmt, Syrk, Trmm, Trsm)
    private val wholeMatrices = EngineImplementation.entries.filter { it.ordinal >= Gemv.ordinal }.toSet()

    /** Owned JVM preferences for the current vector width, or null when the module is unavailable. */
    public val jvmSimd: EngineProfile? get() = simdBackend()?.let {
        jvm(it.minimumVectorLength, SparseTuning.simdIndexedCrossover)
    }

    /** Complete portable coverage with no accelerated preferences. */
    public val portable: EngineProfile = EngineProfile(
        "portable-v1",
        emptyList(),
        evidence = "portable availability floor",
    )

    /** Standard profiles applicable to a target; composition also checks backend availability. */
    public fun forTarget(runtime: String, host: HostPlatform): List<EngineProfile> = listOfNotNull(
        nativeLinuxOneMkl,
        jvmLinuxOneMkl,
        if (runtime == "jvm") jvmSimd else null,
    ).filter { it.appliesTo(runtime, host) }

    /** Builds the owned JVM configuration for the runtime's actual lane width. */
    internal fun jvm(lanes: Int, indexedCrossover: Int): EngineProfile = EngineProfile(
        "jvm-simd-v1",
        listOf(
            EngineRule("jvm-simd", setOf(Iamax), rows = maxOf(lanes, 256), contiguous = true),
            EngineRule("jvm-simd", setOf(Dot, Sum, Asum, Nrm2, Axpy), rows = lanes, contiguous = true),
            EngineRule("jvm-simd", setOf(IndexedDot), rows = maxOf(lanes, indexedCrossover)),
            EngineRule("jvm-simd", setOf(Panels, Products, Triangular)),
        ),
        runtime = "jvm",
        evidence = "koblas-bench sweep and indexed suites; see BACKENDS.md",
    )

    /** Native Intel Linux preferences retained from the i9-12900H benchmark suites. */
    public val nativeLinuxOneMkl: EngineProfile = EngineProfile(
        "native-linux-x64-intel-onemkl-v1",
        listOf(
            EngineRule("onemkl", setOf(Nrm2, Iamax), rows = 128),
            EngineRule("onemkl", setOf(Dot, Asum, Axpy, Scale, Swap, Rot), rows = 64),
            EngineRule("onemkl", halfWork, work = 2048),
            EngineRule("onemkl", wholeMatrices - halfWork, work = 1024),
        ),
        "native",
        intelLinux,
        "i9-12900H; oneMKL 2026.1; Kotlin/Native 2.4.10; koblas-bench Level 1 and dense suites",
    )

    /** JVM oneMKL preferences including operand transfers, for measured Intel Linux targets. */
    public val jvmLinuxOneMkl: EngineProfile = EngineProfile(
        "jvm-linux-x64-intel-onemkl-v1",
        listOf(
            EngineRule("onemkl", levelThree, rows = 128, columns = 128, depth = 128),
            EngineRule("onemkl", levelThree - Gemmt, rows = 64, columns = 64, depth = 64, borrowed = false),
            EngineRule(
                "onemkl",
                setOf(Gemm),
                rows = 16,
                columns = 16,
                depth = 1024,
                maximumDepth = 4096,
                borrowed = false,
            ),
        ),
        "jvm",
        intelLinux,
        "Intel AVX-512; koblas-bench transfer-inclusive suites; see BACKENDS.md",
    )
}

/** Platform code supplies runtime access facts; profile data owns the backend preferences. */
@Suppress("SpreadOperator") // Backend arrays are copied only during immutable engine construction.
internal fun standardEngine(runtime: String, nativeAccess: Boolean = true): KoblasEngine {
    val profiles = StandardEngineProfiles.forTarget(runtime, hostPlatform())
    val rules = profiles.flatMap { it.rules }.filter { nativeAccess || it.backend == "jvm-simd" }
    val vendors = Vendor.entries.mapNotNull { vendor ->
        if (rules.none { it.backend == vendor.vendorName.lowercase() }) {
            null
        } else {
            openBlas(vendor)?.let(EngineBackends::vendor)
        }
    }
    val owned = BuiltinEngines.simd ?: BuiltinEngines.scalar
    if (vendors.isEmpty()) return owned
    return KoblasEngine.combine(
        EngineProfile(profiles.joinToString("+") { it.name }, rules),
        *(vendors + listOfNotNull(EngineBackends.simd)).toTypedArray(),
    )
}
