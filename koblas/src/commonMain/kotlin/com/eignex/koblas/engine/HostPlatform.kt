package com.eignex.koblas.engine

import com.eignex.koblas.internal.engine.cpuClass

/** Operating systems the binding layer distinguishes. */
public enum class OperatingSystem {
    /** Any Linux distribution. */
    Linux,

    /** macOS, where Accelerate is part of the system. */
    MacOs,

    /** Windows, with optional installed BLAS DLLs. */
    Windows,

    /** Anything else, which has no supported vendor. */
    Other,
}

/** Instruction set architectures the binding layer distinguishes. */
public enum class Architecture {
    /** 64-bit x86. */
    X86_64,

    /** 64-bit Arm. */
    Arm64,

    /** Anything else, which has no supported vendor. */
    Other,
}

/** SIMD instruction classes distinguished by measured profiles. */
public enum class CpuClass {
    /** x86-64 without a reported AVX2 extension. */
    X86_64,

    /** x86-64 with AVX2. */
    Avx2,

    /** x86-64 with AVX-512 foundation instructions. */
    Avx512,

    /** 64-bit Arm with its fixed-width platform SIMD. */
    Arm64,

    /** 64-bit Arm with Scalable Vector Extension instructions. */
    Sve,
}

/**
 * The host facts measured profiles match. Resolved once per process; none of these change while it runs.
 */
public class HostPlatform @kotlin.jvm.JvmOverloads constructor(
    /** The operating system. */
    public val operatingSystem: OperatingSystem,
    /** The instruction set architecture. */
    public val architecture: Architecture,
    /** The reported CPU model, or null when it could not be identified. */
    public val cpuModel: String? = null,
    /** Reported SIMD instruction class, or null when it cannot be established. */
    public val cpuClass: CpuClass? = null,
    /** Active SVE vector width in bits, or null when SVE is unavailable or its width is unknown. */
    public val sveVectorBits: Int? = null,
) {
    init {
        if (sveVectorBits != null) {
            require(cpuClass == CpuClass.Sve)
            require(sveVectorBits in 128..2048 && sveVectorBits % 128 == 0)
        }
    }

    override fun toString(): String = "$operatingSystem/$architecture/${cpuModel ?: "unknown"}"
}

/** The platform this process is running on. */
public expect fun hostPlatform(): HostPlatform
