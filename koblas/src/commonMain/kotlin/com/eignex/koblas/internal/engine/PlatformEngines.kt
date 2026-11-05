package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels

/** Platform kernels with eligible whole-call host composition and portable fallbacks. */
internal expect fun platformEngine(): KoblasEngine

internal expect fun simdBackend(): List<EngineKernels>?

internal expect fun vendorEngine(vendor: Vendor): KoblasEngine?

/** One composition path for automatic selection, explicit profiles and benchmark configurations. */
internal fun configuredEngine(
    profile: EngineProfile,
    owned: KoblasEngine? = if (profile.rules.any { it.source == "jvm-simd" }) BuiltinEngines.simd else null,
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine {
    val vendors = Vendor.entries.filter { vendor -> profile.rules.any { it.source == vendor.vendorName.lowercase() } }
        .mapNotNull(openVendor)
    return combine(profile, *(vendors + listOfNotNull(owned)).toTypedArray())
}
