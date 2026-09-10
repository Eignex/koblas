@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, com.eignex.koblas.ExperimentalKoblasApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.engine
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.EOF
import platform.posix.fclose
import platform.posix.fgetc
import platform.posix.fopen
import platform.posix.fputs
import platform.posix.getenv
import platform.posix.mkdir

internal actual fun readTextFile(path: String): String = memScoped {
    val file = fopen(path, "rb") ?: error("cannot open $path")
    try {
        buildString {
            while (true) {
                val byte = fgetc(file)
                if (byte == EOF) break
                append(byte.toChar())
            }
        }
    } finally { fclose(file) }
}

internal actual fun writeTextFile(path: String, text: String) = memScoped {
    val separator = path.lastIndexOf('/')
    if (separator >= 0) {
        var current = if (path.startsWith('/')) "/" else ""
        for (part in path.substring(0, separator).split('/').filter { it.isNotEmpty() }) {
            current = if (current == "/") "/$part" else if (current.isEmpty()) part else "$current/$part"
            mkdir(current, 0x1ffu)
        }
    }
    val file = fopen(path, "wb") ?: error("cannot open $path (create its parent directory first)")
    try { check(fputs(text, file) >= 0) { "write failed for $path" } }
    finally { fclose(file) }
}

internal actual fun resolveEngine(mode: String): Pair<KoblasContext, String> {
    require(mode == "native") { "Native runner requires --mode=native" }
    val provider = requireNotNull(BuiltinKernels.c) { "requested native C engine is unavailable" }
    val engine = provider.engine()
    return engine to "native/${engine.vectorKernels.name}/${engine.sparseKernels.name}/packed-${engine.packedKernels.gemmTileRows}x${engine.packedKernels.gemmTileCols}"
}

internal actual fun runtimeIdentity(): String = "kotlin-native-2.4.10"
internal actual fun environment(name: String): String? = getenv(name)?.toKString()
