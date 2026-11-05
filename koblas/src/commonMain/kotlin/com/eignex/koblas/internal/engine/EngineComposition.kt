package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels

/**
 * Combines available arithmetic with ordered routing preferences and complete portable coverage.
 *
 * Backends named by no applicable rule contribute no arithmetic. Known unavailable built-in backends fall through
 * to later preferences and then portable Kotlin; unknown source identifiers are rejected. This never changes
 * [koblas]. Panel, product and triangular families are selected unconditionally before a
 * schedule starts, because packing and grouping must use the same geometry as the selected arithmetic.
 * Whole-call portable preferences retain the shared schedule with those separately selected families.
 */
internal fun combine(tuning: EngineTuning, vararg backends: KoblasEngine): KoblasEngine {
    require(tuning.rules.none { rule -> !rule.unconditional && rule.implementations.any { it.family } }) {
        "Scheduled kernel families require unconditional selection before packing"
    }
    // Every engine this library builds supplies the kernel seam, which composition binds.
    return CombinedEngine(CompiledRouting(tuning, backends.map { it as EngineKernels }), tuning.name)
}

/** Binds the selected tuning once, opening only the providers it names. */
internal fun standardEngine(
    runtime: EngineRuntime,
    nativeAccess: Boolean = true,
    host: HostPlatform = hostPlatform(),
    owned: KoblasEngine? = BuiltinEngines.simd,
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine {
    val tuning = standardTuning(runtime, host, nativeAccess)
    val vendors = Vendor.entries.filter { vendor -> tuning.rules.any { it.source == vendor.vendorName.lowercase() } }
        .mapNotNull(openVendor)
    return combine(tuning, *(vendors + listOfNotNull(owned)).toTypedArray())
}
