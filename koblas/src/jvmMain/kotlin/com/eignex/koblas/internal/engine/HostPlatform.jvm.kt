package com.eignex.koblas.internal.engine

import java.io.File

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy { detectHost(operatingSystem(), architecture(), ::cpuInfo) }

/** Platform facts read once from JVM properties and Linux CPU metadata where available. */
internal actual fun hostPlatform(): HostPlatform = host

private fun operatingSystem(): OperatingSystem {
    if (System.getProperty("java.vm.name") == "Dalvik") return OperatingSystem.Other
    val name = System.getProperty("os.name").orEmpty()
    return when {
        name.startsWith("Linux", ignoreCase = true) -> OperatingSystem.Linux
        name.startsWith("Mac", ignoreCase = true) -> OperatingSystem.MacOs
        name.startsWith("Windows", ignoreCase = true) -> OperatingSystem.Windows
        else -> OperatingSystem.Other
    }
}

private fun architecture(): Architecture = when (System.getProperty("os.arch")) {
    "amd64", "x86_64" -> Architecture.X86_64
    "aarch64", "arm64" -> Architecture.Arm64
    else -> Architecture.Other
}

/** Linux supplies the model and instruction flags used by calibrated profiles. */
private fun cpuInfo(key: String): String? = try {
    File("/proc/cpuinfo").bufferedReader().use { reader ->
        reader.lineSequence().firstOrNull { it.startsWith(key) }
            ?.substringAfter(':')?.trim()?.takeIf { it.isNotEmpty() }
    }
} catch (_: java.io.IOException) {
    null
}
