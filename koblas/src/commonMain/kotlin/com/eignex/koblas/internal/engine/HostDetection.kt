package com.eignex.koblas.internal.engine

import com.eignex.koblas.engine.*

internal fun cpuClass(architecture: Architecture, flags: String?): CpuClass? = when {
    flags.isNullOrBlank() -> null

    architecture == Architecture.Arm64 -> when {
        "sve" in flags.split(' ') -> CpuClass.Sve
        "asimd" in flags.split(' ') || "neon" in flags.split(' ') -> CpuClass.Arm64
        else -> null
    }

    architecture != Architecture.X86_64 -> null

    "avx512f" in flags.split(' ') -> CpuClass.Avx512

    "avx2" in flags.split(' ') -> CpuClass.Avx2

    else -> CpuClass.X86_64
}

/**
 * Host facts from one platform's probes. Only Linux has a CPU information file to read, and Apple silicon
 * has Advanced SIMD without reporting it anywhere.
 */
internal fun detectHost(
    os: OperatingSystem,
    arch: Architecture,
    cpuInfo: (key: String) -> String?,
    sveVectorBits: () -> Int?,
): HostPlatform {
    val read = { key: String -> if (os == OperatingSystem.Linux) cpuInfo(key) else null }
    val flags = if (os == OperatingSystem.MacOs && arch == Architecture.Arm64) {
        "asimd"
    } else {
        read(if (arch == Architecture.Arm64) "Features" else "flags")
    }
    val instructionClass = cpuClass(arch, flags)
    val bits = if (instructionClass == CpuClass.Sve) sveVectorBits() else null
    return HostPlatform(os, arch, read("model name"), instructionClass, bits)
}

/** Drive rooted and UNC paths are independent of the current directory and the current drive. */
internal fun String.isWindowsAbsolutePath(): Boolean =
    (length >= 3 && this[0].isLetter() && this[1] == ':' && this[2] == '/') ||
        (startsWith("//") && split('/').filter { it.isNotEmpty() }.size >= 2)
