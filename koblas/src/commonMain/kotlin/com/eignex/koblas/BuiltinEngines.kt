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
 * The default uses one platform profile with optional host libraries and portable coverage. JVM profiles
 * also retain [simd] when available. Android and desktop JDKs below 25 use [scalar] without host lookup.
 */
public object BuiltinEngines {
    /** Opens exactly [vendor] when available. Unsupported operations remain unsupported. */
    public fun vendor(vendor: Vendor): KoblasEngine? = vendorEngine(vendor)

    /**
     * Resolves a profile's built-in sources and retains owned SIMD preferences beneath its rules.
     * Missing optional libraries retain portable or SIMD coverage; unknown source identifiers fail.
     * The caller checks [EngineProfile.appliesTo] before choosing this configuration.
     */
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
        simdBackend()?.let { KoblasEngine.combine(requireNotNull(StandardEngineProfiles.jvmSimd), *it.toTypedArray()) }
    }
}

internal expect fun simdBackend(): List<KoblasEngine>?

internal expect fun vendorEngine(vendor: Vendor): KoblasEngine?

/** One composition path for automatic selection, explicit profiles and benchmark configurations. */
internal fun configuredEngine(
    profile: EngineProfile,
    owned: KoblasEngine? = BuiltinEngines.simd,
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine {
    val vendors = Vendor.entries.filter { vendor -> profile.rules.any { it.source == vendor.vendorName.lowercase() } }
        .mapNotNull(openVendor)
    val rules = profile.rules + if (owned != null && profile.rules.none { it.source == "jvm-simd" }) {
        StandardEngineProfiles.jvmSimd?.rules.orEmpty()
    } else {
        emptyList()
    }
    val combined = KoblasEngine.combine(
        EngineProfile(profile.name, rules),
        *(vendors + listOfNotNull(owned)).toTypedArray(),
    )
    if (vendors.isEmpty() && (owned == null || profile.rules.isEmpty())) return owned ?: BuiltinEngines.scalar
    return combined
}
