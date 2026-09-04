package com.eignex.koblas.sparse.host

internal actual fun parentDirectory(path: String): String? =
    path.substringBeforeLast('/', missingDelimiterValue = "").ifEmpty { null }
