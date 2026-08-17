package com.eignex.koblas

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 */
internal object ConfigurationKeys {
    /** Pins backend selection to one [BackendNames] value instead of taking the highest priority offered. */
    const val BACKEND_PROPERTY = "koblas.backend"

    /** Thread count for a host OpenBLAS, which koblas otherwise pins to one. */
    const val OPENBLAS_THREADS_PROPERTY = "koblas.openblas.threads"

    /** OpenBLAS's own thread count. When it is set koblas leaves the threading to the library. */
    const val OPENBLAS_THREADS_ENV = "OPENBLAS_NUM_THREADS"

    /** Prefix of the per-level [DispatchThresholds] overrides, completed by [DispatchLevel.key]. */
    const val DISPATCH_PROPERTY_PREFIX = "koblas.dispatch."

    /** Prefix of the same overrides read from the environment, completed by the [DispatchLevel] name. */
    const val DISPATCH_ENV_PREFIX = "KOBLAS_DISPATCH_"
}
