@file:OptIn(kotlin.experimental.ExperimentalNativeApi::class)

package com.eignex.koblas.vendor

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    HostPlatform(operatingSystem(), architecture(), cpuVendor())
}

/** The platform this process is running on, which a Native target largely knows at compile time. */
public actual fun hostPlatform(): HostPlatform = host

/** The target's own identity, which Kotlin/Native knows at compile time and needs no probe for. */
private fun operatingSystem(): OperatingSystem = when (Platform.osFamily) {
    OsFamily.LINUX -> OperatingSystem.Linux
    OsFamily.MACOSX -> OperatingSystem.MacOs
    else -> OperatingSystem.Other
}

private fun architecture(): Architecture = when (Platform.cpuArchitecture) {
    CpuArchitecture.X64 -> Architecture.X86_64
    CpuArchitecture.ARM64 -> Architecture.Arm64
    else -> Architecture.Other
}

/**
 * The CPU manufacturer, read from `/proc/cpuinfo` on Linux and left unknown elsewhere.
 *
 * Nothing else needs it: macOS takes Accelerate and Arm takes ArmPL without consulting the manufacturer, so it
 * only ever chooses between AOCL and oneMKL on 64-bit x86.
 */
@OptIn(ExperimentalForeignApi::class)
private fun cpuVendor(): CpuVendor {
    if (operatingSystem() != OperatingSystem.Linux) return CpuVendor.Unknown
    val file = fopen("/proc/cpuinfo", "r") ?: return CpuVendor.Unknown
    try {
        memScoped {
            val line = allocArray<kotlinx.cinterop.ByteVar>(LINE_BUFFER)
            while (fgets(line, LINE_BUFFER, file) != null) {
                val text = line.toKString()
                if (!text.startsWith("vendor_id")) continue
                val value = text.substringAfter(':', "").trim()
                return when {
                    value.equals("GenuineIntel", ignoreCase = true) -> CpuVendor.Intel
                    value.equals("AuthenticAMD", ignoreCase = true) -> CpuVendor.Amd
                    else -> CpuVendor.Unknown
                }
            }
        }
    } finally {
        fclose(file)
    }
    return CpuVendor.Unknown
}

private const val LINE_BUFFER = 512
