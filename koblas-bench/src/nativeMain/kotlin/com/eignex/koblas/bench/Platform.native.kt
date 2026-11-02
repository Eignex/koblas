@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import com.eignex.koblas.vendor.hostPlatform
import com.eignex.koblas.vendor.OperatingSystem
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.getenv
import platform.posix.EOF
import platform.posix.fclose
import platform.posix.fgetc
import platform.posix.fputs

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
    val engine = if (mode == "native-default") profileEngine("native") ?: koblas else BuiltinEngines.scalar
    return engine to engineIdentity(mode, engine)
}

internal actual fun runtimeIdentity(): String = "kotlin-native-${KotlinVersion.CURRENT}"

internal actual fun engineProfilePath(): String? = getenv("KOBLAS_BENCH_ENGINE_PROFILE")?.toKString()
