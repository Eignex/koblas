package com.eignex.koblas.vendor

import java.nio.file.Files
import java.nio.file.Path

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    HostPlatform(operatingSystem(), architecture(), cpuVendor())
}

/** The platform this process is running on, read from the JVM's own system properties. */
public actual fun hostPlatform(): HostPlatform = host

private fun operatingSystem(): OperatingSystem {
    val name = System.getProperty("os.name").orEmpty()
    return when {
        name.startsWith("Linux", ignoreCase = true) -> OperatingSystem.Linux
        name.startsWith("Mac", ignoreCase = true) -> OperatingSystem.MacOs
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
 * Linux reports it in `/proc/cpuinfo`. Everywhere else it stays [CpuVendor.Unknown], which is not a gap: macOS
 * takes Accelerate and Arm takes ArmPL without consulting the manufacturer at all.
 */
private fun cpuVendor(): CpuVendor {
    if (operatingSystem() != OperatingSystem.Linux) return CpuVendor.Unknown
    val line = try {
        Files.newBufferedReader(Path.of("/proc/cpuinfo")).use { reader ->
            reader.lineSequence().firstOrNull { it.startsWith("vendor_id") }
        }
    } catch (_: java.io.IOException) {
        null
    } ?: return CpuVendor.Unknown
    val value = line.substringAfter(':', "").trim()
    return when {
        value.equals("GenuineIntel", ignoreCase = true) -> CpuVendor.Intel
        value.equals("AuthenticAMD", ignoreCase = true) -> CpuVendor.Amd
        else -> CpuVendor.Unknown
    }
}
