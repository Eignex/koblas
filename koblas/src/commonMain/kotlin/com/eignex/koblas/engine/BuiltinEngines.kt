package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.portable.PortableEngine

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * The default uses one platform profile with optional host libraries and portable coverage. JVM profiles
 * also retain [simd] when available. Android and desktop JDKs below 25 use [scalar] without host lookup.
 */
public object BuiltinEngines {
    /** Opens exactly [vendor] when available. Unsupported operations remain unsupported. */
    @kotlin.jvm.JvmStatic
    public fun vendor(vendor: Vendor): KoblasEngine? = vendorEngine(vendor)

    /**
     * Resolves exactly the profile's declared routing preferences and built-in sources.
     * Missing optional libraries retain the profile's other choices and portable coverage.
     * Unknown source identifiers fail.
     * The caller checks [EngineProfile.appliesTo] before choosing this configuration.
     */
    @kotlin.jvm.JvmStatic
    public fun configured(profile: EngineProfile): KoblasEngine = configuredEngine(profile)

    /** Portable Kotlin across all three levels. */
    @get:kotlin.jvm.JvmStatic
    public val scalar: KoblasEngine get() = PortableEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * On the JVM, [koblas] retains these kernels when available, including in its host composition.
     * A window too short or too strided for a vector body runs a portable fallback.
     */
    @get:kotlin.jvm.JvmStatic
    public val simd: KoblasEngine? by lazy {
        simdBackend()?.let { parts ->
            val rule = EngineRule("jvm-simd", parts.flatMapTo(mutableSetOf()) { it.implementations })
            combine(EngineProfile("jvm-simd", listOf(rule)), *parts.toTypedArray())
        }
    }
}
