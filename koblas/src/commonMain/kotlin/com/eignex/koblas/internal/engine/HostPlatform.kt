package com.eignex.koblas.internal.engine

internal enum class EngineRuntime { Jvm, Native }
internal enum class OperatingSystem { Linux, MacOs, Windows, Other }
internal enum class Architecture { X86_64, Arm64, Other }
internal enum class CpuClass { X86_64, Avx2, Avx512, Arm64, Sve }

/** Process facts used by internal tuning and benchmark provenance. */
internal data class HostPlatform(
    val operatingSystem: OperatingSystem,
    val architecture: Architecture,
    val cpuModel: String? = null,
    val cpuClass: CpuClass? = null,
    val sveVectorBits: Int? = null,
) {
    init {
        if (sveVectorBits != null) {
            require(cpuClass == CpuClass.Sve)
            require(sveVectorBits in 128..2048 && sveVectorBits % 128 == 0)
        }
    }
}

internal expect fun hostPlatform(): HostPlatform
