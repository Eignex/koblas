package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

internal expect fun platformEngine(): KoblasEngine
internal expect fun platformEngineSetting(): String?
internal expect fun simdBackend(): BuiltinEngine?
internal expect fun vendorEngine(vendor: Vendor): KoblasEngine?

/** Try compatible libraries in order and keep the first available one for the life of the engine. */
internal fun standardEngine(
    runtime: EngineRuntime,
    nativeAccess: Boolean = true,
    host: HostPlatform = hostPlatform(),
    owned: KoblasEngine? = BuiltinEngines.simd,
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine {
    if (host.architecture == Architecture.Other) return ScalarEngine
    val selected = if (nativeAccess) vendorOrder(runtime, host).firstNotNullOfOrNull(openVendor) else null
    if (selected == null && owned == null) return ScalarEngine
    return BuiltinEngine(
        "auto-${selected?.id ?: "owned"}",
        (owned as? BuiltinEngine)?.kernels ?: KernelSet(),
        selected as? EngineKernels,
        EngineTuning(runtime, host),
    )
}
