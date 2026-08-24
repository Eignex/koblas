package com.eignex.koblas.internal.backend

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 */
internal object ConfigurationKeys {
    /** Pins dense backend selection to one [BackendNames] value instead of taking the highest priority offered. */
    const val DENSE_BACKEND_PROPERTY = "koblas.dense.backend"

    /** Pins sparse backend selection to one [BackendNames] value instead of taking the highest priority offered. */
    const val SPARSE_BACKEND_PROPERTY = "koblas.sparse.backend"

    /** An absolute path to the host OpenBLAS, overriding the deployment lookup chain. */
    val OPENBLAS_PATH = LibraryPathKeys("koblas.openblas.path", "KOBLAS_OPENBLAS_PATH")

    /** An absolute path to the host LAPACKE, for a host that keeps it outside its OpenBLAS. */
    val LAPACKE_PATH = LibraryPathKeys("koblas.lapacke.path", "KOBLAS_LAPACKE_PATH")

    /** An absolute path to the host KLU. */
    val KLU_PATH = LibraryPathKeys("koblas.klu.path", "KOBLAS_KLU_PATH")

    /** An absolute path to the host UMFPACK. */
    val UMFPACK_PATH = LibraryPathKeys("koblas.umfpack.path", "KOBLAS_UMFPACK_PATH")
}

/** The system property and the environment variable a deployment can point one library path at. */
internal class LibraryPathKeys(val property: String, val environment: String)
