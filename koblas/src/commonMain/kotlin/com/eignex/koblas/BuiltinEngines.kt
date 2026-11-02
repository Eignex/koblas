@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.vendor.Vendor

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine by lazy { platformEngine() }

/** Platform kernels with eligible whole-call host composition and portable fallbacks. */
internal expect fun platformEngine(): KoblasEngine

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * The JVM default retains [simd] when available and [scalar] otherwise as its owned kernels, and may
 * compose eligible oneMKL matrix calls when native access is enabled. Native has no SIMD
 * engine and uses an applicable measured host profile, or [scalar] when none applies.
 * Android and desktop JDKs below 25 use [scalar] without resolving a host library.
 */
public object BuiltinEngines {
    /** Opens exactly [vendor] when available. Unsupported operations remain unsupported. */
    public fun vendor(vendor: Vendor): KoblasEngine? = vendorEngine(vendor)

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
        simdBackend()?.let { KoblasEngine.combine(requireNotNull(StandardEngineProfiles.jvmSimd), *it.toTypedArray()) }
    }
}

internal expect fun simdBackend(): List<KoblasEngine>?

internal expect fun vendorEngine(vendor: Vendor): KoblasEngine?
