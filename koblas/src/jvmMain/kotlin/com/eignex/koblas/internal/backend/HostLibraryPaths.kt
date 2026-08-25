package com.eignex.koblas.internal.backend

/**
 * The path a deployment configured for one library, the system property ahead of the environment variable.
 * A relative path is ignored rather than resolved, since what it would resolve against is the working
 * directory of whoever launched the process.
 */
internal fun libraryPath(keys: LibraryPathKeys): String? = System.getProperty(keys.property)?.takeIf(::isAbsolutePath)
    ?: System.getenv(keys.environment)?.takeIf(::isAbsolutePath)

internal fun isAbsolutePath(path: String): Boolean = java.nio.file.Path.of(path).isAbsolute
