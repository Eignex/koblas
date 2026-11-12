package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.KernelProvider
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
        "jvm-scalar" -> KoblasEngine.scalar
        "jvm-simd" -> requireNotNull(KoblasEngine.simd) {
            "requested jvm-simd engine is unavailable; launch with jdk.incubator.vector"
        }
        // What an ordinary caller gets, which is a policy rather than an exact arm: its panels and kernels
        // are whatever this platform selected, and the engine name in each row says which those are.
        "jvm-default", "jvm-default-no-simd" -> koblas.also {
            check(mode != "jvm-default-no-simd" || KoblasEngine.simd == null) {
                "jvm-default-no-simd must launch without jdk.incubator.vector"
            }
        }
        else -> automaticVendorFromMode(mode)?.let { vendor ->
            check(KoblasEngine.simd == null) { "$mode must launch without jdk.incubator.vector" }
            val host = requireNotNull(KoblasEngine.exactVendor(vendor)) { "Requested ${vendor.vendorName} is unavailable" }
            val selected = standardEngine(EngineRuntime.Jvm, nativeAccess = KoblasEngine::class.java.module.isNativeAccessEnabled, owned = null) { candidate ->
                (host as KernelProvider).takeIf { candidate == vendor }
            }
            check(selected.providers.any { it.vendor == vendor }) { "Requested vendor is incompatible with this host" }
            selected
        } ?: error("unknown JVM mode $mode")
    }
    return engine to engineIdentity(mode, engine)
}

internal actual fun runtimeIdentity(): String = "kotlin-${KotlinVersion.CURRENT}/jvm/${System.getProperty("java.vendor")}/${System.getProperty("java.version")}".replace(',', '_')

internal actual fun RoutingKey.provider(engine: KoblasEngine): String =
    engine.providerFor(operation, rows, columns, depth, contiguous, borrowed)
