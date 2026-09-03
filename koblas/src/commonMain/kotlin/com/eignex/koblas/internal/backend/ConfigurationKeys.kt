package com.eignex.koblas.internal.backend

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 *
 * The keys that pin one backend half sit on [BackendSlot] beside the half they select, so a half cannot be
 * added without them.
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

    /** An absolute path to the host KLU. */
    val KLU_PATH = LibraryPathKeys("koblas.klu.path", "KOBLAS_KLU_PATH")

    /** An absolute path to the host UMFPACK. */
    val UMFPACK_PATH = LibraryPathKeys("koblas.umfpack.path", "KOBLAS_UMFPACK_PATH")

    /** An absolute path to the host CHOLMOD. */
    val CHOLMOD_PATH = LibraryPathKeys("koblas.cholmod.path", "KOBLAS_CHOLMOD_PATH")

    /** An absolute path to a BASICLU exporting koblas's bridge entry points, ahead of the bundled build. */
    val BASICLU_PATH = LibraryPathKeys("koblas.basiclu.path", "KOBLAS_BASICLU_PATH")

    /** An absolute path to a build of koblas's HFactor bridge, ahead of the bundled one. */
    val HFACTOR_PATH = LibraryPathKeys("koblas.hfactor.path", "KOBLAS_HFACTOR_PATH")
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
