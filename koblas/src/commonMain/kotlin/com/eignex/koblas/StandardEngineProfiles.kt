package com.eignex.koblas

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.vendor.*

/** Complete platform compositions with provisional vendor cutoffs awaiting pre-merge calibration. */
public object StandardEngineProfiles {
    private val levelThree = setOf(Gemm, Gemmt, Symm, Syrk, Syr2k, Trmm, Trsm)
    private val halfWork = setOf(Syr, Trmv, TrmvTransposed, Trsv, TrsvTransposed, Gemmt, Syrk, Trmm, Trsm)
    private val wholeMatrices = EngineImplementation.entries.filter { it.matrix }.toSet()

    /** Owned JVM preferences for the current vector width, or null when the module is unavailable. */
    internal val jvmSimd: EngineProfile? by lazy {
        simdBackend()?.let { backend ->
            val lanes = backend.first().minimumLength(Dot)
            EngineProfile(
                "jvm-simd-v1",
                listOf(
                    EngineRule("jvm-simd", setOf(Iamax), rows = maxOf(lanes, 256)..Int.MAX_VALUE, contiguous = true),
                    EngineRule(
                        "jvm-simd",
                        setOf(Dot, Sum, Asum, Nrm2, Axpy),
                        rows = lanes..Int.MAX_VALUE,
                        contiguous = true,
                    ),
                    EngineRule(
                        "jvm-simd",
                        setOf(IndexedDot),
                        rows = maxOf(lanes, 16)..Int.MAX_VALUE,
                    ),
                    EngineRule("jvm-simd", setOf(Panels, Products, Triangular)),
                ),
                runtime = EngineRuntime.Jvm,
                evidence = "koblas-bench sweep and indexed suites; see BACKENDS.md",
            )
        }
    }

    private val portable = EngineProfile("portable", emptyList())
    private val profiles by lazy {
        EngineRuntime.entries.flatMap { runtime ->
            listOf(OperatingSystem.Linux, OperatingSystem.MacOs).flatMap { os ->
                (
                    if (os == OperatingSystem.MacOs) {
                        listOf(
                            Architecture.Arm64,
                        )
                    } else {
                        listOf(Architecture.X86_64, Architecture.Arm64)
                    }
                    ).map { arch ->
                    val vendors = if (os == OperatingSystem.MacOs) {
                        listOf(Vendor.Accelerate)
                    } else {
                        buildList {
                            if (arch == Architecture.X86_64) add(Vendor.OneMkl)
                            add(Vendor.OpenBlas)
                            if (runtime == EngineRuntime.Jvm) {
                                add(
                                    if (arch == Architecture.X86_64) Vendor.Aocl else Vendor.ArmPl,
                                )
                            }
                        }
                    }
                    EngineProfile(
                        "${runtime.name.lowercase()}-$os-$arch-v1",
                        vendors.flatMap { vendorRules(it, runtime) } +
                            if (runtime == EngineRuntime.Jvm) jvmSimd?.rules.orEmpty() else emptyList(),
                        runtime,
                        HostPlatform(os, arch),
                        "Provisional estimates; measure each vendor and target before merge",
                    )
                }
            }
        }
    }

    /** One complete prebuilt profile; unavailable providers fall through to owned SIMD or portable arithmetic. */
    public fun forTarget(runtime: EngineRuntime, host: HostPlatform, nativeAccess: Boolean = true): EngineProfile =
        if (nativeAccess) {
            profiles.firstOrNull {
                it.appliesTo(
                    runtime,
                    host,
                )
            } ?: bestEffortProfile(runtime, host, profiles) ?: portable
        } else {
            if (runtime == EngineRuntime.Jvm) jvmSimd ?: portable else portable
        }

    private fun vendorRules(vendor: Vendor, runtime: EngineRuntime): List<EngineRule> {
        val source = vendor.vendorName.lowercase()
        if (runtime == EngineRuntime.Jvm) {
            val order = if (vendor == Vendor.OneMkl || vendor == Vendor.Accelerate) 64 else 128
            fun region(
                operations: Set<EngineImplementation>,
                minimum: Int,
                borrowed: Boolean? = null,
                depth: IntRange? = null,
            ): EngineRule {
                val extent = minimum..Int.MAX_VALUE
                return EngineRule(source, operations, extent, extent, depth ?: extent, borrowed = borrowed)
            }
            return buildList {
                add(region(levelThree, order * 2))
                add(region(levelThree - Gemmt, order, borrowed = false))
                if (vendor == Vendor.OneMkl) add(region(setOf(Gemm), 16, borrowed = false, depth = 1024..4096))
            }
        }
        val length = if (vendor == Vendor.OpenBlas) 128 else 64
        val work = when (vendor) {
            Vendor.Accelerate -> 512L
            Vendor.OneMkl -> 1024L
            else -> 2048L
        }
        return listOf(
            EngineRule(source, setOf(Nrm2, Iamax), rows = length * 2..Int.MAX_VALUE),
            EngineRule(source, setOf(Dot, Asum, Axpy, Scale, Swap, Rot), rows = length..Int.MAX_VALUE),
            EngineRule(source, halfWork, work = work * 2),
            EngineRule(source, wholeMatrices - halfWork, work = work),
        )
    }
}

/** Borrow performance preferences while keeping runtime bindings and available arithmetic independent. */
internal fun bestEffortProfile(
    runtime: EngineRuntime,
    host: HostPlatform,
    profiles: List<EngineProfile>,
): EngineProfile? {
    if (host.operatingSystem == OperatingSystem.Other || host.architecture == Architecture.Other) return null
    val vendors = when {
        host.operatingSystem == OperatingSystem.MacOs -> setOf(Vendor.Accelerate)
        host.architecture == Architecture.X86_64 -> setOf(Vendor.OneMkl, Vendor.OpenBlas, Vendor.Aocl)
        else -> setOf(Vendor.OpenBlas, Vendor.ArmPl)
    }
    val allowed = vendors.map { it.vendorName.lowercase() }
    fun closest(vendor: Boolean): EngineProfile? = profiles.filter { profile ->
        profile.runtime == runtime && profile.target != null && profile.rules.any {
            if (vendor) it.source in allowed else it.source == "jvm-simd"
        }
    }.maxByOrNull { profile ->
        val target = requireNotNull(profile.target)
        (if (vendor && target.operatingSystem == host.operatingSystem) 100 else 0) +
            (if (target.architecture == host.architecture) 10 else 0) +
            (if (target.cpuClass == host.cpuClass) 2 else 0) +
            (if (target.sveVectorBits == host.sveVectorBits) 1 else 0)
    }
    val vendor = closest(true)
    val owned = if (runtime == EngineRuntime.Jvm) closest(false) else null
    if (vendor == null && owned == null) return null
    return EngineProfile(
        "best-effort-$runtime-${host.operatingSystem}-${host.architecture}",
        vendor?.rules.orEmpty().filter { it.source in allowed } +
            owned?.rules.orEmpty().filter { it.source == "jvm-simd" },
        runtime,
        HostPlatform(host.operatingSystem, host.architecture),
        "Unmeasured best effort; vendor preferences from ${vendor?.name}; owned preferences from ${owned?.name}",
    )
}

/** Platform code supplies access facts; target data and available providers determine the composition. */
internal fun standardEngine(
    runtime: EngineRuntime,
    nativeAccess: Boolean = true,
    host: HostPlatform = hostPlatform(),
    owned: KoblasEngine? = BuiltinEngines.simd,
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine = configuredEngine(StandardEngineProfiles.forTarget(runtime, host, nativeAccess), owned, openVendor)
