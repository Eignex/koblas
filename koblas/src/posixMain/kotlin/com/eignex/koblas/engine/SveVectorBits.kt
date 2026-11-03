@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.engine

import kotlinx.cinterop.CFunction
import kotlinx.cinterop.invoke
import kotlinx.cinterop.reinterpret
import platform.posix.RTLD_NOW
import platform.posix.dlclose
import platform.posix.dlopen
import platform.posix.dlsym

internal actual fun sveVectorBits(): Int? {
    val process = dlopen(null, RTLD_NOW) ?: return null
    try {
        val prctl = dlsym(
            process,
            "prctl",
        )?.reinterpret<CFunction<(Int, ULong, ULong, ULong, ULong) -> Int>>() ?: return null
        // PR_SVE_GET_VL and PR_SVE_VL_LEN_MASK are newer than the Native compiler's Linux sysroot.
        val value = prctl(51, 0UL, 0UL, 0UL, 0UL)
        return if (value < 0) null else ((value and 0xffff) * 8).takeIf { it in 128..2048 && it % 128 == 0 }
    } finally {
        dlclose(process)
    }
}
