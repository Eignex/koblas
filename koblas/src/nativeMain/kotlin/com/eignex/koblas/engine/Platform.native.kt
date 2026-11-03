@file:OptIn(kotlin.experimental.ExperimentalNativeApi::class)

package com.eignex.koblas.engine

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    val os = operatingSystem()
    val arch = architecture()
    val flags = if (os == OperatingSystem.MacOs && arch == Architecture.Arm64) {
        "asimd"
    } else {
        cpuInfo(if (arch == Architecture.Arm64) "Features" else "flags")
    }
    val instructionClass = cpuClass(arch, flags)
    HostPlatform(
        os,
        arch,
        cpuInfo("model name"),
        instructionClass,
        if (instructionClass == CpuClass.Sve) sveVectorBits() else null,
    )
}

/** Linux reports the current thread's SVE width without changing its vector configuration. */
internal expect fun sveVectorBits(): Int?

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

/** Linux supplies the model and instruction flags used by calibrated profiles. */
@OptIn(ExperimentalForeignApi::class)
private fun cpuInfo(key: String): String? {
    if (operatingSystem() != OperatingSystem.Linux) return null
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
