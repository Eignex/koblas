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
            checkNotNull(
                configuredVendorEngine(vendor, openVendor),
            ) { "Koblas engine '$name' is unavailable on this platform" }
        }
    }
}

/** Named selection keeps exactly one host, with portable coverage for operations it does not export. */
internal fun configuredVendorEngine(
    vendor: Vendor,
    openVendor: (Vendor) -> KernelProvider? = ::vendorEngine,
): KernelProvider? = openVendor(vendor)?.let { BuiltinEngine(vendor.vendorName.lowercase(), host = it) }

/** Try compatible libraries in order and keep the first available one for the life of the engine. */
internal fun standardEngine(
    runtime: EngineRuntime,
    nativeAccess: Boolean = true,
    host: HostPlatform = hostPlatform(),
    owned: BuiltinEngine? = simdEngine,
    openVendor: (Vendor) -> KernelProvider? = ::vendorEngine,
): KernelProvider {
    if (host.architecture == Architecture.Other) return ScalarEngine
    val selected = if (nativeAccess) vendorOrder(runtime, host).firstNotNullOfOrNull(openVendor) else null
    if (selected == null && owned == null) return ScalarEngine
    return BuiltinEngine(
        "auto-${selected?.id ?: "owned"}",
        owned?.roles ?: KernelSet(),
        selected,
        EngineTuning(runtime, host, selected?.vendor, ownedSimd = owned != null),
    )
}

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
            listOf(Vendor.OpenBlas, Vendor.OneMkl)
        }
        ) + listOfNotNull(Vendor.Aocl.takeIf { runtime == EngineRuntime.Jvm })
}

/** Drive rooted and UNC paths are independent of the current directory and the current drive. */
internal fun String.isWindowsAbsolutePath(): Boolean =
    (length >= 3 && this[0].isLetter() && this[1] == ':' && this[2] == '/') ||
        (startsWith("//") && split('/').filter { it.isNotEmpty() }.size >= 2)

/**
 * Stands in for the user's home directory inside a vendor's candidate library path.
 *
 * Outside [Vendor] because its entries are constructed before a companion of their own class is initialized,
 * and outside the public package so the JVM facade a file-level constant needs is not published.
 */
internal const val USER_HOME: String = "{home}"

internal expect fun platformEngine(): KernelProvider
internal expect fun platformEngineSetting(): String?
internal expect fun simdBackend(): BuiltinEngine?
internal expect fun vendorEngine(vendor: Vendor): KernelProvider?

internal val simdEngine: BuiltinEngine? by lazy { simdBackend() }
