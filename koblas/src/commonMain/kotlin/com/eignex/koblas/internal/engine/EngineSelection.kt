package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** Explicit names select only the requested source, with portable coverage for missing vendor operations. */
internal fun engineFromSetting(
    setting: String?,
    automatic: () -> KernelProvider = ::platformEngine,
    simd: () -> BuiltinEngine? = { simdEngine },
    openVendor: (Vendor) -> KernelProvider? = ::vendorEngine,
): KernelProvider {
    val name = setting?.trim()?.lowercase() ?: "auto"
    return when (name) {
        "auto" -> automatic()

        "scalar" -> ScalarEngine

        "simd" -> checkNotNull(simd()) { "Koblas engine 'simd' is unavailable; enable the JVM Vector API module" }

        else -> {
            val vendor = requireNotNull(Vendor.entries.firstOrNull { it.vendorName.lowercase() == name }) {
                "Unknown Koblas engine '$setting'; expected auto, scalar, simd, " +
                    Vendor.entries.joinToString { it.vendorName.lowercase() }
            }
            val engine = checkNotNull(openVendor(vendor)) { "Koblas engine '$name' is unavailable on this platform" }
            BuiltinEngine(name, host = engine)
        }
    }
}
