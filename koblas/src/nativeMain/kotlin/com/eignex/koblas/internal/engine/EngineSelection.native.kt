@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import kotlinx.cinterop.toKString
import platform.posix.getenv

internal actual fun simdBackend(): BuiltinEngine? = null

/** Keep one host vendor with portable coverage when no library is available. */
internal actual fun platformEngine(): KernelProvider = standardEngine(EngineRuntime.Native)

internal actual fun platformEngineSetting(): String? = getenv("KOBLAS_ENGINE")?.toKString()

internal actual fun vendorEngine(vendor: Vendor): KernelProvider? =
    com.eignex.koblas.internal.host.NativeVendorBlas.open(vendor)
