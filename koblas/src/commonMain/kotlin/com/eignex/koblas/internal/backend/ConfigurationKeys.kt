package com.eignex.koblas.internal.backend

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 */
internal object ConfigurationKeys {
    /** Pins dense backend selection to one [BackendNames] value instead of taking the highest priority offered. */
    const val DENSE_BACKEND_PROPERTY = "koblas.dense.backend"

    /** The environment variable behind [DENSE_BACKEND_PROPERTY], for a deployment that sets no properties. */
    const val DENSE_BACKEND_ENVIRONMENT = "KOBLAS_DENSE_BACKEND"

    /** Pins sparse backend selection to one [BackendNames] value instead of taking the highest priority offered. */
    const val SPARSE_BACKEND_PROPERTY = "koblas.sparse.backend"

    /** The environment variable behind [SPARSE_BACKEND_PROPERTY]. */
    const val SPARSE_BACKEND_ENVIRONMENT = "KOBLAS_SPARSE_BACKEND"

    /** An absolute path to the library exporting `cblas_*`, overriding the deployment lookup chain. */
    val CBLAS_PATH = LibraryPathKeys("koblas.cblas.path", "KOBLAS_CBLAS_PATH")

    /** An absolute path to the library exporting `LAPACKE_*`, for a host that keeps it outside its CBLAS. */
    val LAPACKE_PATH = LibraryPathKeys("koblas.lapacke.path", "KOBLAS_LAPACKE_PATH")

    /** An absolute path to the host KLU. */
    val KLU_PATH = LibraryPathKeys("koblas.klu.path", "KOBLAS_KLU_PATH")

    /** An absolute path to the host UMFPACK. */
    val UMFPACK_PATH = LibraryPathKeys("koblas.umfpack.path", "KOBLAS_UMFPACK_PATH")
}

/** The system property and the environment variable a deployment can point one library path at. */
internal class LibraryPathKeys(val property: String, val environment: String)

/**
 * The backend pin a deployment asked for, [property] ahead of [environment]. Blank counts as unset, since a
 * variable exported empty is a deployment that meant to clear the pin rather than one asking for a backend
 * no name matches.
 */
internal fun pinnedBackend(property: String?, environment: String?): String? =
    (property ?: environment)?.trim()?.takeIf { it.isNotEmpty() }
