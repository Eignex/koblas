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
    HostPlatform(operatingSystem(), architecture(), cpuModel())
}

/** The platform this process is running on, which a Native target largely knows at compile time. */
public actual fun hostPlatform(): HostPlatform = host

/** The target's own identity, which Kotlin/Native knows at compile time and needs no probe for. */
private fun operatingSystem(): OperatingSystem = when (Platform.osFamily) {
    OsFamily.LINUX -> OperatingSystem.Linux
    OsFamily.MACOSX -> OperatingSystem.MacOs
    OsFamily.WINDOWS -> OperatingSystem.Windows
    else -> OperatingSystem.Other
}

private fun architecture(): Architecture = when (Platform.cpuArchitecture) {
    CpuArchitecture.X64 -> Architecture.X86_64
    CpuArchitecture.ARM64 -> Architecture.Arm64
    else -> Architecture.Other
}

/** Unknown CPU models retain portable coverage instead of borrowing another model's measurements. */
@OptIn(ExperimentalForeignApi::class)
private fun cpuModel(): String? {
    if (operatingSystem() != OperatingSystem.Linux) return null
    val file = fopen("/proc/cpuinfo", "r") ?: return null
    try {
        memScoped {
            val line = allocArray<kotlinx.cinterop.ByteVar>(LINE_BUFFER)
            while (fgets(line, LINE_BUFFER, file) != null) {
                val text = line.toKString()
                if (!text.startsWith("model name")) continue
                return text.substringAfter(':').trim().takeIf { it.isNotEmpty() }
            }
        }
    } finally {
        fclose(file)
    }
    return null
}

private const val LINE_BUFFER = 512
