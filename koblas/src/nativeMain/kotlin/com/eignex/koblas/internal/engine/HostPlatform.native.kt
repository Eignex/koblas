@file:OptIn(kotlin.experimental.ExperimentalNativeApi::class)

package com.eignex.koblas.internal.engine

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy { detectHost(operatingSystem(), architecture(), ::cpuInfo) }

/** The platform this process is running on, which a Native target largely knows at compile time. */
internal actual fun hostPlatform(): HostPlatform = host

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

/** Linux supplies the model and instruction flags used by measured cutoffs. */
@OptIn(ExperimentalForeignApi::class)
private fun cpuInfo(key: String): String? {
    val file = fopen("/proc/cpuinfo", "r") ?: return null
    try {
        memScoped {
            val line = allocArray<kotlinx.cinterop.ByteVar>(LINE_BUFFER)
            while (fgets(line, LINE_BUFFER, file) != null) {
                val text = line.toKString()
                if (!text.startsWith(key)) continue
                return text.substringAfter(':').trim().takeIf { it.isNotEmpty() }
            }
        }
    } finally {
        fclose(file)
    }
    return null
}

private const val LINE_BUFFER = 4096
