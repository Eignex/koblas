package com.eignex.koblas.internal.backend

import java.nio.file.InvalidPathException
import java.nio.file.Path

/** The configured absolute path, then the platform lookup names for one native library. */
internal fun nativeLibraryPaths(
    property: String,
    environment: String,
    defaults: List<String>,
    propertyValue: String? = System.getProperty(property),
    environmentValue: String? = System.getenv(environment),
): List<String> = listOfNotNull(
    propertyValue?.takeIf(::isAbsolutePath) ?: environmentValue?.takeIf(::isAbsolutePath),
) + defaults

private fun isAbsolutePath(value: String): Boolean = try {
    Path.of(value).isAbsolute
} catch (_: InvalidPathException) {
    false
}
