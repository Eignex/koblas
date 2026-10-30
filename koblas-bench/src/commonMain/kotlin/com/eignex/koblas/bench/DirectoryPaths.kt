package com.eignex.koblas.bench

/** Parents to create in order, preserving POSIX, Windows drive and UNC roots. */
internal fun parentDirectories(path: String, windows: Boolean = false): List<String> {
    val normalized = if (windows) path.replace('\\', '/') else path
    val separator = normalized.lastIndexOf('/')
    if (separator < 0) return emptyList()
    val parent = normalized.substring(0, separator)
    val parts = parent.split('/').filter { it.isNotEmpty() }
    val rootParts = when {
        windows && normalized.startsWith("//") -> 2
        windows && normalized.length >= 3 && normalized[1] == ':' && normalized[2] == '/' -> 1
        else -> 0
    }
    var current = when {
        windows && normalized.startsWith("//") -> "//" + parts.take(rootParts).joinToString("/")
        rootParts == 1 -> parts.first()
        normalized.startsWith('/') -> ""
        else -> ""
    }
    val absolute = normalized.startsWith('/') || rootParts > 0
    return parts.drop(rootParts).map { part ->
        current = if (current.isEmpty() && !absolute) part else "$current/$part"
        current
    }
}
