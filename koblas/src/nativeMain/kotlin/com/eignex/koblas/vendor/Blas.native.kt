package com.eignex.koblas.vendor

import com.eignex.koblas.internal.host.NativeVendorBlas

/** Opens the preferred available vendor for this host, or null when none is installed. */
public actual fun openBlas(only: Vendor?): Blas? {
    val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
    return candidates.firstNotNullOfOrNull(NativeVendorBlas::open)
}
