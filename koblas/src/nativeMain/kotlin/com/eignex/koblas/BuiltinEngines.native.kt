package com.eignex.koblas

import com.eignex.koblas.vendor.Vendor

internal actual fun simdBackend(): List<KoblasEngine>? = null

/** Select an applicable measured profile once, with portable coverage when none matches. */
internal actual fun platformEngine(): KoblasEngine = standardEngine("native")

internal actual fun vendorEngine(vendor: Vendor): KoblasEngine? =
    com.eignex.koblas.internal.host.NativeVendorBlas.open(vendor)
