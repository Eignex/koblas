@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.EOF
import platform.posix.fclose
import platform.posix.fgetc
import platform.posix.fopen
import platform.posix.fputs

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
            createDirectory(current)
        }
    }
    val file = fopen(path, "wb") ?: error("cannot open $path (create its parent directory first)")
    try { check(fputs(text, file) >= 0) { "write failed for $path" } }
    finally { fclose(file) }
}

/**
 * The two Koblas arms this runtime has, which are different engines rather than one engine named twice.
 *
 * `native` is the exact portable one: common Kotlin at every level, resolving no library, which is what a
 * comparison against this platform's own arithmetic needs. `native-default` is what an ordinary call gets,
 * which on this platform is a policy that hands a whole dense Level 2 or 3 call to an installed library once
 * there is enough arithmetic to pay for reaching it. The generic entry points have no engine to be told, so
 * their rows belong on the second and are declined on the first.
 *
 * The identity carries the library where the default composed one, so a report says which binary the rows
 * that reached it were produced by rather than leaving the arm's name to imply it.
 */
internal actual fun resolveEngine(mode: String): Pair<KoblasEngine, String> {
    // Kotlin/Native has one Level 1 selection, the scalar kernels; the Vector API is a JVM module.
    require(mode == "native" || mode == "native-default") {
        "Native runner requires --mode=native or --mode=native-default"
    }
    val engine = if (mode == "native-default") koblas else BuiltinEngines.scalar
    val host = engine.vendor?.let {
        "/${it.vendor.vendorName}/${it.libraryPath}/${it.version}/threads=${it.threadEvidence.label}"
    } ?: ""
    return engine to "$mode/${engine.name}$host"
}

internal actual fun runtimeIdentity(): String = "kotlin-native-2.4.10"
