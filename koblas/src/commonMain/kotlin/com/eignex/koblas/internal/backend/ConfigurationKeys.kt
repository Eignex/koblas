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
}
