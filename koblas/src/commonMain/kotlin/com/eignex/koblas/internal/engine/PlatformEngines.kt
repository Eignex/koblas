package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

internal expect fun platformEngine(): KernelProvider
internal expect fun platformEngineSetting(): String?
internal expect fun simdBackend(): BuiltinEngine?
internal expect fun vendorEngine(vendor: Vendor): KernelProvider?

internal val simdEngine: BuiltinEngine? by lazy { simdBackend() }

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
        EngineTuning(runtime, host, selected?.vendor),
    )
}
