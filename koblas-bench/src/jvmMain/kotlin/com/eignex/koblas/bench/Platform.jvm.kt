package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import java.nio.file.Files
import java.nio.file.Path

internal actual fun readTextFile(path: String): String = Files.readString(Path.of(path))

internal actual fun writeTextFile(path: String, text: String) {
    val target = Path.of(path)
    target.parent?.let(Files::createDirectories)
    Files.writeString(target, text)
}

internal actual fun resolveEngine(mode: String): Pair<KoblasEngine, String> {
    val engine = when (mode) {
        "jvm-scalar" -> BuiltinEngines.scalar
        "jvm-simd" -> requireNotNull(BuiltinEngines.simd) {
            "requested jvm-simd engine is unavailable; launch with jdk.incubator.vector"
        }
        else -> error("unknown JVM mode $mode")
    }
    return engine to "$mode/${engine.vectorKernels.name}/${engine.sparseKernels.name}/${engine.denseImplementation}"
}

internal actual fun runtimeIdentity(): String = "kotlin-2.4.10/jvm/${System.getProperty("java.vendor")}/${System.getProperty("java.version")}".replace(',', '_')
