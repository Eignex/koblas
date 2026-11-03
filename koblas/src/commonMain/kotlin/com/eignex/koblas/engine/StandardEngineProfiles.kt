package com.eignex.koblas.engine

import com.eignex.koblas.*

/** Complete target configurations with measured profiles and documented best-effort preferences. */
public object StandardEngineProfiles {
    private val portable = EngineProfile("portable", emptyList())
    private val profiles = standardProfiles

    /** One prebuilt profile, borrowing documented best-effort preferences when the target is unmeasured. */
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
}

/** Borrow performance preferences while keeping runtime bindings and available arithmetic independent. */
internal fun bestEffortProfile(
    runtime: EngineRuntime,
    host: HostPlatform,
    profiles: List<EngineProfile>,
): EngineProfile? {
    if (host.architecture == Architecture.Other) return null
    val vendors = when {
        host.operatingSystem == OperatingSystem.Other -> emptySet()
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
    val donors = listOfNotNull(vendor, owned).distinct()
    if (donors.isEmpty()) return null
    return EngineProfile(
        "best-effort-$runtime-${host.operatingSystem}-${host.architecture}-${donors.joinToString("+") { it.name }}",
        vendor?.rules.orEmpty().filter { it.source in allowed } +
            owned?.rules.orEmpty().filter { it.source == "jvm-simd" },
        runtime,
        HostPlatform(host.operatingSystem, host.architecture),
        "Unmeasured best effort; " + listOfNotNull(
            vendor?.let { "vendor preferences from ${it.name}" },
            owned?.let { "owned preferences from ${it.name}" },
        ).joinToString("; "),
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
