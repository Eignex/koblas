@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.providerFor
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.EOF
import platform.posix.fclose
import platform.posix.fgetc
import platform.posix.fputs
import platform.posix.getenv

internal actual fun readTextFile(path: String): String = memScoped {
    val file = openTextFile(path, "rb") ?: error("cannot open $path")
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
    parentDirectories(path,
        windows = hostPlatform().operatingSystem == OperatingSystem.Windows).forEach(::createDirectory)
    val file = openTextFile(path, "wb") ?: error("cannot open $path (create its parent directory first)")
    try { check(fputs(text, file) >= 0) { "write failed for $path" } }
    finally { fclose(file) }
}

/** Exact portable and standard/explicit combined arms, with selected binding provenance. */
internal actual fun resolveEngine(mode: String): Pair<KoblasEngine, String> {
    require(mode == "native" || mode == "native-default") {
        "Native runner requires --mode=native or --mode=native-default"
    }
    val engine = if (mode == "native-default") koblas else KoblasEngine.scalar
    return engine to engineIdentity(mode, engine)
}

internal actual fun runtimeIdentity(): String = "kotlin-native-${KotlinVersion.CURRENT}"

internal actual fun RoutingKey.provider(engine: KoblasEngine): String =
    engine.providerFor(operation, rows, columns, depth, contiguous, borrowed)
