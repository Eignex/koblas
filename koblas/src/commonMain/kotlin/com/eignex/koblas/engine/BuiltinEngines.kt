package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * Application startup selection belongs to [configureKoblas]. These exact engines are useful for
 * conformance tests and comparisons; vendor engines support only their exported operations.
 */
public object BuiltinEngines {
    /** Opens exactly [vendor] when available. Unsupported operations remain unsupported. */
    @kotlin.jvm.JvmStatic
    public fun vendor(vendor: Vendor): KoblasEngine? = vendorEngine(vendor)

    /** Portable Kotlin across all three levels. */
    @get:kotlin.jvm.JvmStatic
    public val scalar: KoblasEngine get() = ScalarEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * On the JVM, [koblas] retains these kernels when available, alongside its selected host library.
     * A window too short or too strided for a vector body runs a portable fallback.
     */
    @get:kotlin.jvm.JvmStatic
    public val simd: KoblasEngine? by lazy { simdBackend() }
}
