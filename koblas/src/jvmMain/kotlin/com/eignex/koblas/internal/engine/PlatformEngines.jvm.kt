package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.jvm.JvmBackend
import com.eignex.koblas.internal.kernels.EngineKernels

/** Owned JVM kernels, with eligible host calls when native access and a vendor are available. */
internal actual fun platformEngine(): KoblasEngine = JvmBackend.default

internal actual fun simdBackend(): List<EngineKernels>? = JvmBackend.contributions

internal actual fun vendorEngine(vendor: Vendor): KoblasEngine? = JvmBackend.open(vendor)
