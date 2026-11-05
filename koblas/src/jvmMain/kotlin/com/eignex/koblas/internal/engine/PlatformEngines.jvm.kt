package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.Vendor
import com.eignex.koblas.internal.jvm.JvmBackend
import com.eignex.koblas.internal.kernels.EngineKernels

/** Owned JVM kernels, with eligible host calls when native access and a vendor are available. */
internal actual fun platformEngine(): EngineKernels = JvmBackend.default

internal actual fun platformEngineSetting(): String? = System.getProperty(
    "koblas.engine",
) ?: System.getenv("KOBLAS_ENGINE")

internal actual fun simdBackend(): BuiltinEngine? = JvmBackend.simd

internal actual fun vendorEngine(vendor: Vendor): EngineKernels? = JvmBackend.open(vendor)
