package com.eignex.koblas.internal.kernels

import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT

/** OS capability queries use native buffers and ordinary FFM transitions. */
internal actual object NativeProbe {
    private val handle = JvmNativeLibrary.library?.handle(
        "koblas_probe_v1",
        FfmLibrary.intOf(ADDRESS, ADDRESS),
        critical = false,
    )

    actual fun query(kind: Int, index: Int, kernelId: Int): IntArray? {
        val probe = handle ?: return null
        return Arena.ofConfined().use { arena ->
            val request = arena.allocate(32, 4)
            val result = arena.allocate(256, 4)
            request.set(JAVA_INT, 0, 1)
            request.set(JAVA_INT, 4, 32)
            request.set(JAVA_INT, 8, kind)
            request.set(JAVA_INT, 12, index)
            request.set(JAVA_INT, 16, kernelId)
            result.set(JAVA_INT, 0, 1)
            result.set(JAVA_INT, 4, 256)
            val status = probe.invokeExact(request, result) as Int
            if (status != 0 && kind == 1) return null
            check(status == 0) { "native probe rejected: $status" }
            result.toArray(JAVA_INT)
        }
    }
}
