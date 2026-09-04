package com.eignex.koblas.sparse.host

import java.io.File

internal actual fun parentDirectory(path: String): String? = File(path).parent
