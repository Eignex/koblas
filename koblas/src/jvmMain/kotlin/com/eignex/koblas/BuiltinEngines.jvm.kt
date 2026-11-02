package com.eignex.koblas

import com.eignex.koblas.vendor.Vendor

/** Owned JVM kernels, with eligible host calls when native access and a measured vendor are available. */
internal actual fun platformEngine(): KoblasEngine = JvmBackend.default

internal actual fun simdBackend(): List<KoblasEngine>? = JvmBackend.contributions

internal actual fun vendorEngine(vendor: Vendor): KoblasEngine? = JvmBackend.open(vendor)
