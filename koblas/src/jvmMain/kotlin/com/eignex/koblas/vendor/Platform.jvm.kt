package com.eignex.koblas.vendor

import com.sun.management.HotSpotDiagnosticMXBean
import java.io.File
import java.lang.management.ManagementFactory

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
    val bits = if (instructionClass == CpuClass.Sve) sveVectorBits() else null
    HostPlatform(os, arch, cpuInfo("model name"), instructionClass, bits)
}

/** Platform facts read once from JVM properties and Linux CPU metadata where available. */
public actual fun hostPlatform(): HostPlatform = host

/** The scalar arm omits the Vector API module but must report the same active VM width. */
private fun sveVectorBits(): Int? = try {
    val vm = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java)
    if (vm?.getVMOption("UseSVE")?.value?.toIntOrNull()?.let { it > 0 } == true) {
        vm.getVMOption("MaxVectorSize").value.toIntOrNull()?.times(8)
    } else {
        null
    }
} catch (_: IllegalArgumentException) {
    null
}

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

/** Linux supplies the model and instruction flags used by calibrated profiles. */
private fun cpuInfo(key: String): String? {
    if (operatingSystem() != OperatingSystem.Linux) return null
    return try {
        File("/proc/cpuinfo").bufferedReader().use { reader ->
            reader.lineSequence().firstOrNull { it.startsWith(key) }
                ?.substringAfter(':')?.trim()?.takeIf { it.isNotEmpty() }
        }
    } catch (_: java.io.IOException) {
        null
    }
}
