package com.eignex.koblas.vendor

import java.io.File

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    HostPlatform(operatingSystem(), architecture(), cpuModel())
}

/** Platform facts read once from JVM properties and Linux CPU metadata where available. */
public actual fun hostPlatform(): HostPlatform = host

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

/** Unknown CPU models retain portable coverage instead of borrowing another model's measurements. */
private fun cpuModel(): String? {
    if (operatingSystem() != OperatingSystem.Linux) return null
    return try {
        File("/proc/cpuinfo").bufferedReader().use { reader ->
            reader.lineSequence().firstOrNull { it.startsWith("model name") }
                ?.substringAfter(':')?.trim()?.takeIf { it.isNotEmpty() }
        }
    } catch (_: java.io.IOException) {
        null
    }
}
