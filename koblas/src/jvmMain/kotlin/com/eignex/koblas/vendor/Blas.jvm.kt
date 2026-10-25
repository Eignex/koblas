@file:JvmName("JvmVendorBlasKt")

package com.eignex.koblas.vendor

import com.eignex.koblas.JvmBackend

/** Opens an optional desktop FFM binding, or returns null on portable runtimes. */
public actual fun openBlas(only: Vendor?): Blas? = JvmBackend.openBlas(only)
