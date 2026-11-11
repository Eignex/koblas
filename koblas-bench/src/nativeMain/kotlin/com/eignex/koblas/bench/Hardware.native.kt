@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.internal.engine.OperatingSystem
import com.eignex.koblas.internal.engine.hostPlatform
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen

internal actual fun cpuModel(): String? {
    if (hostPlatform().operatingSystem != OperatingSystem.Linux) return null
    val file = fopen("/proc/cpuinfo", "r") ?: return null
    try {
        memScoped {
            val line = allocArray<ByteVar>(LINE_BUFFER)
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

private const val LINE_BUFFER = 4096
