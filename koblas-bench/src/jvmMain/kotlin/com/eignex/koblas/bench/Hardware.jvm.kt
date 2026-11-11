package com.eignex.koblas.bench

import com.eignex.koblas.internal.engine.CpuClass
import com.eignex.koblas.internal.engine.OperatingSystem
import com.eignex.koblas.internal.engine.hostPlatform
import com.sun.management.HotSpotDiagnosticMXBean
import java.io.File
import java.lang.management.ManagementFactory

internal actual fun cpuModel(): String? {
    if (hostPlatform().operatingSystem != OperatingSystem.Linux) return null
    return try {
        File("/proc/cpuinfo").bufferedReader().use { reader ->
            reader.lineSequence().firstOrNull { it.startsWith("model name") }
                ?.substringAfter(':')?.trim()?.takeIf { it.isNotEmpty() }
        }
    } catch (_: java.io.IOException) {
        null
    }
}

/** The scalar arm omits the Vector API module but must report the same active VM width. */
internal actual fun sveVectorBits(): Int? {
    if (hostPlatform().cpuClass != CpuClass.Sve) return null
    return try {
        val vm = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java)
        if (vm?.getVMOption("UseSVE")?.value?.toIntOrNull()?.let { it > 0 } == true) {
            vm.getVMOption("MaxVectorSize").value.toIntOrNull()?.times(8)
        } else {
            null
        }
    } catch (_: IllegalArgumentException) {
        null
    }
}
