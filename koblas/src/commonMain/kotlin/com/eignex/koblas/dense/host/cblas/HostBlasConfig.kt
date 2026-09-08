package com.eignex.koblas.dense.host.cblas

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers. Such a build
 * exports the same unsuffixed symbols as LP64 and then reads the wrong halves of every dimension.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    config.split(' ', '\t', '\n').any { it == "USE64BITINT" || it == "INTERFACE64" }

/** Whether [config] identifies an OpenBLAS build whose CBLAS integer ABI is the LP64 one koblas binds. */
internal fun isLp64OpenBlas(config: String): Boolean =
    config.split(' ', '\t', '\n').any { it == "OpenBLAS" } && !isIlp64OpenBlas(config)

/** Names used by the platform loader to locate a host OpenBLAS. */
internal val OPENBLAS_SONAMES = listOf(
    "libopenblas.so.0",
    "libopenblas.so",
    "libopenblas.dylib",
    "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
    "/usr/local/opt/openblas/lib/libopenblas.dylib",
    "openblas.dll",
)

/** Numerical and execution policy shared by host and bundled OpenBLAS providers. */
public data class OpenBlasOptions(
    /** OpenBLAS thread count; null leaves the library's process-wide setting unchanged. */
    val threadCount: Int? = 1,
) {
    init {
        require(threadCount == null || threadCount > 0) { "threadCount must be positive" }
    }
}

/** Policy for one OpenBLAS backend instance. */
public data class HostBlasConfig(
    /** An absolute OpenBLAS library path, or the deployment lookup chain when null. */
    val libraryPath: String? = null,
    /** OpenBLAS thread count; setting it, including to one, requires a higher JVM thread-memory cap. */
    val threadCount: Int? = 1,
) {
    /** Creates a deployment-discovered host configuration from shared [options]. */
    public constructor(options: OpenBlasOptions) : this(null, options.threadCount)

    /** Creates a host configuration from library locations and shared [options]. */
    public constructor(
        libraryPath: String?,
        options: OpenBlasOptions,
    ) : this(
        libraryPath,
        options.threadCount,
    )

    /** Numerical and execution policy, independent of the library locations. */
    public val options: OpenBlasOptions
        get() = OpenBlasOptions(threadCount)

    init {
        require(threadCount == null || threadCount > 0) { "threadCount must be positive" }
    }
}

internal fun OpenBlasOptions.metadataOptions(effectiveThreadCount: Int? = threadCount): Map<String, String> = buildMap {
    effectiveThreadCount?.let { put("threadCount", it.toString()) }
}
