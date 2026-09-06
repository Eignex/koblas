package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64BundledBackend

/** One JVM system property, or null off the JVM, where there are none. */
internal expect fun systemPropertyOrNull(name: String): String?

/** One environment variable, or null on a target that cannot read the environment. */
internal expect fun environmentVariableOrNull(name: String): String?

/**
 * The path a deployment configured for one library, the system property ahead of the environment variable.
 * A relative path is ignored rather than resolved, since what it would resolve against is the working
 * directory of whoever launched the process.
 */
internal fun libraryPath(keys: LibraryPathKeys): String? =
    (systemPropertyOrNull(keys.property) ?: environmentVariableOrNull(keys.environment))?.takeIf(::isAbsolutePath)

/** The backend pin a deployment asked for, read the same way as a library path. */
internal fun requestedBackend(keys: BackendSelectionKeys): String? =
    pinnedBackend(systemPropertyOrNull(keys.property), environmentVariableOrNull(keys.environment))

/** The backend pins for every semantic role, read independently. */
internal fun requestedBackends(): Map<BackendSlot, String?> =
    BackendSlot.entries.associateWith { requestedBackend(it.selectionKeys) }

/**
 * Whether a configured path is absolute. This is the shape the keys accept rather than full path semantics:
 * a leading separator, or a drive letter for the Windows spelling.
 */
internal fun isAbsolutePath(path: String): Boolean = path.startsWith('/') || path.startsWith('\\') ||
    (path.length >= 3 && path[1] == ':' && (path[2] == '/' || path[2] == '\\'))

/** Whether [provider] answers to [requested], by its own name or by the one a bundled build is configured under. */
internal fun matchesRequested(provider: Backend, requested: String): Boolean =
    provider.name == requested || (provider as? F64BundledBackend)?.canonicalName == requested
