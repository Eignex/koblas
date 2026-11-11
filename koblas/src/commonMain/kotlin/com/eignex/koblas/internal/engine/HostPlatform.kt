package com.eignex.koblas.internal.engine

internal enum class EngineRuntime { Jvm, Native }
internal enum class OperatingSystem { Linux, MacOs, Windows, Other }
internal enum class Architecture { X86_64, Arm64, Other }
internal enum class CpuClass { X86_64, Avx2, Avx512, Arm64, Sve }

/** Process facts used by engine selection and internal tuning. */
internal data class HostPlatform(
    val operatingSystem: OperatingSystem,
    val architecture: Architecture,
    val cpuClass: CpuClass? = null,
)

internal expect fun hostPlatform(): HostPlatform

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
internal fun detectHost(os: OperatingSystem, arch: Architecture, cpuInfo: (key: String) -> String?): HostPlatform {
    val read = { key: String -> if (os == OperatingSystem.Linux) cpuInfo(key) else null }
    val flags = if (os == OperatingSystem.MacOs && arch == Architecture.Arm64) {
        "asimd"
    } else {
        read(if (arch == Architecture.Arm64) "Features" else "flags")
    }
    return HostPlatform(os, arch, cpuClass(arch, flags))
}
