package com.eignex.koblas.internal.backend

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 *
 * The keys that pin one backend half sit on [BackendSlot] beside the half they select, so a half cannot be
 * added without them. The dense cache block sizes and dispatch crossovers are keyed the same way and sit on
 * their entries in `DenseTuning`, where each key belongs beside the default it overrides and the
 * measurement that chose that default.
 */
internal object ConfigurationKeys {
    /** Selects indexed JVM Vector API stores for sparse kernels. */
    val JVM_VECTOR_SCATTER = JvmVectorScatterKeys(
        "koblas.jvm.vector.scatter",
        "KOBLAS_JVM_VECTOR_SCATTER",
    )

    /** An absolute path to the library exporting `cblas_*`, overriding the deployment lookup chain. */
    val CBLAS_PATH = LibraryPathKeys("koblas.cblas.path", "KOBLAS_CBLAS_PATH")

    /** An absolute path to the library exporting `LAPACKE_*`, for a host that keeps it outside its CBLAS. */
    val LAPACKE_PATH = LibraryPathKeys("koblas.lapacke.path", "KOBLAS_LAPACKE_PATH")

    /** An absolute path to a build of koblas's HFactor bridge, ahead of the bundled one. */
    val HFACTOR_PATH = LibraryPathKeys("koblas.hfactor.path", "KOBLAS_HFACTOR_PATH")

    /**
     * The keys a deployment points at each library, by the name its provider is configured under.
     *
     * Declared beside the keys so a library added here joins discovery's configured-path precedence without
     * a second table to remember: a bundled provider whose library a deployment configured steps aside, and
     * one missing from this map silently would not.
     */
    val LIBRARY_PATHS: Map<String, List<LibraryPathKeys>> = mapOf(
        BackendNames.OPENBLAS to listOf(CBLAS_PATH, LAPACKE_PATH),
        BackendNames.HFACTOR to listOf(HFACTOR_PATH),
    )
}

/** The system property and the environment variable a deployment can point one library path at. */
internal class LibraryPathKeys(val property: String, val environment: String)

/** The system property and environment variable that pin one semantic backend role. */
internal class BackendSelectionKeys(val property: String, val environment: String)

/** The JVM property and environment variable controlling indexed JVM Vector API stores. */
internal class JvmVectorScatterKeys(val property: String, val environment: String)

/**
 * The backend pin a deployment asked for, [property] ahead of [environment]. Blank counts as unset, since a
 * variable exported empty is a deployment that meant to clear the pin rather than one asking for a backend
 * no name matches.
 */
internal fun pinnedBackend(property: String?, environment: String?): String? =
    (property ?: environment)?.trim()?.takeIf { it.isNotEmpty() }
