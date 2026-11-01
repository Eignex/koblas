package com.eignex.koblas

/** Owned JVM kernels, with eligible host calls when native access and a measured vendor are available. */
internal actual fun platformEngine(): KoblasEngine = JvmBackend.default

internal actual fun simdBackend(): EngineBackend? = JvmBackend.backend
