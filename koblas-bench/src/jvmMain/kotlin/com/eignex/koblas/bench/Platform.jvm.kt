package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.engine
import java.nio.file.Files
import java.nio.file.Path

internal actual fun readTextFile(path: String): String = Files.readString(Path.of(path))

internal actual fun writeTextFile(path: String, text: String) {
    val target = Path.of(path)
    target.parent?.let(Files::createDirectories)
    Files.writeString(target, text)
}

internal actual fun resolveEngine(mode: String): Pair<KoblasContext, String> {
    val provider = when (mode) {
        "jvm-c" -> requireNotNull(BuiltinKernels.c) { "requested jvm-c engine is unavailable" }
        "jvm-simd" -> requireNotNull(BuiltinKernels.simd) { "requested jvm-simd engine is unavailable; launch with jdk.incubator.vector" }
        else -> error("JVM runner cannot execute mode '$mode'")
    }
    val engine = provider.engine()
    return engine to "$mode/${engine.vectorKernels.name}/${engine.sparseKernels.name}/packed-${engine.packedKernels.gemmTileRows}x${engine.packedKernels.gemmTileCols}"
}

internal actual fun runtimeIdentity(): String = "kotlin-2.4.10/jvm/${System.getProperty("java.vendor")}/${System.getProperty("java.version")}".replace(',', '_')
internal actual fun environment(name: String): String? = System.getenv(name)
