package com.eignex.koblas

import com.eignex.koblas.vendor.Vendor

internal actual fun simdBackend(): List<KoblasEngine>? = null

/** Select one platform profile once, with portable coverage when no library is available. */
internal actual fun platformEngine(): KoblasEngine = standardEngine(EngineRuntime.Native)

internal actual fun vendorEngine(vendor: Vendor): KoblasEngine? =
    com.eignex.koblas.internal.host.NativeVendorBlas.open(vendor)
