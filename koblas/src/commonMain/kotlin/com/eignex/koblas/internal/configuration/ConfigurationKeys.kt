package com.eignex.koblas.internal.configuration

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 *
 * Dense cache block sizes and dispatch crossovers are keyed beside their defaults in `DenseTuning`.
 */
internal object ConfigurationKeys {
    /** Selects indexed JVM Vector API stores for sparse kernels. */
    val JVM_VECTOR_SCATTER = JvmVectorScatterKeys(
        "koblas.jvm.vector.scatter",
        "KOBLAS_JVM_VECTOR_SCATTER",
    )
}

/** The JVM property and environment variable controlling indexed JVM Vector API stores. */
internal class JvmVectorScatterKeys(val property: String, val environment: String)
