@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlinx.cinterop.*
import platform.posix.*

private fun clock(clock: Int): Long = memScoped {
    val time = alloc<timespec>()
    check(clock_gettime(clock, time.ptr) == 0)
    time.tv_sec * 1_000_000_000L + time.tv_nsec
}

public fun scalDiagnostic() {
    val vectors = requireNotNull(BuiltinEngines.c).vectorKernels
    println("n,offset,alignment_bytes,mode,sample,operations,cpu_ns,wall_ns,cpu_ns_per_op,wall_ns_per_op")
    var sink = 0.0
    for (n in intArrayOf(64, 256, 4096, 65536)) {
        val initial = Fixtures.vector(n, 1)
        val x = DoubleArray(n + 4)
        for (offset in 0..3) for (mode in 0..1) {
            initial.copyInto(x, offset)
            val alignment = x.usePinned { (it.addressOf(offset).rawValue.toLong() and 31).toInt() }
            val batch = 1_048_576 / n
            for (sample in -2..7) {
                var cpu = 0L
                var wall = 0L
                repeat(16) {
                    val w0 = clock(CLOCK_MONOTONIC)
                    val c0 = clock(CLOCK_THREAD_CPUTIME_ID)
                    repeat(batch) {
                        if (mode == 1) initial.copyInto(x, offset)
                        vectors.scale(x, offset, if (mode == 0) -1.0 else 0.875, n)
                        sink += x[offset] + x[offset + n - 1]
                    }
                    cpu += clock(CLOCK_THREAD_CPUTIME_ID) - c0
                    wall += clock(CLOCK_MONOTONIC) - w0
                }
                if (sample > 0) {
                    val operations = batch * 16
                    println("$n,$offset,$alignment,${if (mode == 0) "arithmetic" else "reset"},$sample,$operations,$cpu,$wall,${cpu.toDouble() / operations},${wall.toDouble() / operations}")
                }
            }
        }
    }
    check(sink.isFinite())
}
