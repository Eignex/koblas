package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Kernels

/** What every backend reports about itself. */
public interface Backend {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    public val name: String

    /**
     * Relative preference among the backends offered for one half ([F64Blas], [F64Kernels] or a
     * sparse counterpart). [registerBackend] picks the highest; the portable reference is 0.
     */
    public val priority: Int get() = 0

    /**
     * Whether this is koblas's own implementation rather than a binding to a host library. The compiled-in
     * SIMD kernels are portable however fast they are; only something calling out counts as accelerated.
     */
    public val isPortable: Boolean get() = false

    /**
     * Whether this backend can do work on this host. koblas's own implementations always can, so the default
     * is true; a binding reports whether the library it calls resolved.
     *
     * Each half answers for itself, since a host can provide OpenBLAS without HFactor. Registration does not
     * consult this: koblas registers a binding on a bare library lookup and
     * lets the binding fall back per call, so a registered backend may still report false here.
     * Read it to report what a host offers, or before installing one explicitly.
     */
    public val isAvailable: Boolean get() = true

    /**
     * Why this backend is unavailable, or null when it is available or cannot say.
     *
     * A binding that failed to resolve knows something a caller cannot recover from [isAvailable] alone: the
     * library was missing, or it loaded but a symbol was not there. Reporting it here rather than on each
     * concrete binding is what lets a caller ask any half the same question instead of downcasting to the
     * one type that happens to answer it.
     */
    public val unavailableReason: String? get() = null
}

/** Runtime facts a backend can report without requiring callers to parse [Backend.name]. */
public data class BackendMetadata(
    /** Library or implementation version, when the provider can determine it. */
    val version: String? = null,
    /** Integer ABI used at the native boundary, such as `"LP64"`, when applicable and known. */
    val integerAbi: String? = null,
    /** Effective threading mode, when the provider can determine it. */
    val threading: String? = null,
    /** Effective numerical and dispatch options, keyed by stable option names. */
    val options: Map<String, String> = emptyMap(),
)

/** Optional capability for a [Backend] that can report structured runtime facts. */
public interface BackendMetadataProvider {
    /** The structured facts this backend knows about itself. */
    public val backendMetadata: BackendMetadata
}

/**
 * The priority every host binding koblas ships registers at. A third-party backend is unprobed, and an ILP64
 * OpenBLAS exports identical symbols while computing wrong answers, caught by reading `openblas_get_config`.
 */
public const val HOST_BACKEND_PRIORITY: Int = 100

/**
 * A portable backend whose kernels a context may replace.
 *
 * The reference backends read the process default kernels when constructed without kernels of their own, so
 * a context built with different ones rebinds them onto its own. Being rebindable is a property of the type
 * rather than an identity against the shared singletons, so a caller who constructs their own stock portable
 * backend gets the context's kernels too, and a portable default added later joins in without the builder
 * having to learn about it.
 */
public interface F64RebindableBackend : Backend {
    /** Whether this backend carries kernels of its own, which a context must leave alone. */
    public val hasOwnKernels: Boolean
}

/**
 * A provider carrying a bundled build of a library koblas also binds.
 *
 * Such a provider answers to two names: its own, which says where the library came from, and the
 * [canonicalName] a deployment configures. Declaring the second is what lets the registry match a pin and
 * let a configured library take precedence, rather than reading the distinction out of the provider's name.
 */
public interface F64BundledBackend : Backend {
    /** The name a caller configures this library under, which this provider also answers to. */
    public val canonicalName: String
}
