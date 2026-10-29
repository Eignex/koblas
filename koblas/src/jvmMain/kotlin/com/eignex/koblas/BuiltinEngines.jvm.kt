package com.eignex.koblas

/** JVM built-in engines. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy { KoblasEngine.portable() }

    /** Vector API kernels on desktop JDK 25 or later, or null on portable runtimes. */
    @get:JvmStatic
    public actual val simd: KoblasEngine? get() = JvmBackend.simd
}

/** Owned JVM kernels, with eligible host calls when native access and a measured vendor are available. */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = JvmBackend.default
