package com.eignex.koblas.engine

import com.eignex.koblas.*

internal actual fun simdBackend(): List<EngineKernels>? = null

/** Select one platform profile once, with portable coverage when no library is available. */
internal actual fun platformEngine(): KoblasEngine = standardEngine(EngineRuntime.Native)

internal actual fun vendorEngine(vendor: Vendor): KoblasEngine? =
    com.eignex.koblas.internal.host.NativeVendorBlas.open(vendor)
