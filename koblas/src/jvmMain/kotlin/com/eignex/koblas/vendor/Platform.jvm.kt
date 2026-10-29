package com.eignex.koblas.vendor

import java.io.File

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    HostPlatform(operatingSystem(), architecture(), cpuVendor())
}

/** The platform this process is running on, read from the JVM's own system properties. */
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

/**
 * The CPU manufacturer, which only ever refines the order within one architecture.
 *
 * Linux reports it in `/proc/cpuinfo`; Windows includes it in `PROCESSOR_IDENTIFIER`.
 * Elsewhere it stays [CpuVendor.Unknown]; macOS and Arm selection do not consult the manufacturer.
 */
private fun cpuVendor(): CpuVendor {
    if (operatingSystem() == OperatingSystem.Windows) {
        return cpuVendorFromIdentifier(System.getenv("PROCESSOR_IDENTIFIER"))
    }
    if (operatingSystem() != OperatingSystem.Linux) return CpuVendor.Unknown
    val line = try {
        File("/proc/cpuinfo").bufferedReader().use { reader ->
            reader.lineSequence().firstOrNull { it.startsWith("vendor_id") }
        }
    } catch (_: java.io.IOException) {
        null
    } ?: return CpuVendor.Unknown
    return cpuVendorFromIdentifier(line.substringAfter(':', "").trim())
}
