package com.eignex.koblas.internal.kernels

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.NativeVariant
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Uninstrumented checks of warmed exact native calls and thread-local result ownership. */
internal object JvmNativeRuntimeCheck {
    @Volatile
    private var sink = 0.0

    @JvmStatic
    fun main(args: Array<String>) {
        check(BuiltinEngines.nativeVariants.isNotEmpty())
        val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean
        bean.isThreadAllocatedMemoryEnabled = true
        for (variant in BuiltinEngines.nativeVariants) {
            val bindings = JvmCKernelBindings(variant)
            repeat(4) { runDots(bindings, 20_000) }
            val before = bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
            runDots(bindings, 20_000)
            val allocated = bean.getThreadAllocatedBytes(Thread.currentThread().threadId()) - before
            check(allocated <= 4096) { "$variant allocated $allocated bytes in warmed calls" }
            println("$variant warmed allocation: $allocated bytes for 20000 calls")
        }
        val executor = Executors.newFixedThreadPool(4)
        try {
            val futures = (0 until 4).map { index ->
                executor.submit {
                    val variant = BuiltinEngines.nativeVariants[index % BuiltinEngines.nativeVariants.size]
                    checkThread(variant, index)
                }
            }
            futures.forEach { it.get(10, TimeUnit.SECONDS) }
        } finally {
            executor.shutdownNow()
        }
        println("concurrent exact calls and read-only thread probes passed")
    }

    private val left = DoubleArray(65) { it * 0.125 }
    private val right = DoubleArray(65) { 1.0 }

    private fun runDots(bindings: JvmCKernelBindings, iterations: Int) {
        var sum = 0.0
        repeat(iterations) { sum += bindings.denseDot(left, 0, right, 0, left.size) }
        sink = sum
    }

    private fun checkThread(variant: NativeVariant, index: Int) {
        val bindings = JvmCKernelBindings(variant)
        val input = DoubleArray(65) { (index + 1).toDouble() }
        val output = DoubleArray(65)
        val expected = 65.0 * (index + 1) * (index + 1)
        repeat(1000) {
            check(bindings.denseDot(input, 0, input, 0, 65) == expected)
            bindings.denseAxpy(output, 0, 1.0, input, 0, 65)
            val context = NativeRecords.context(checkNotNull(NativeProbe.query(3)))
            check(context.readyProcess == 0 && context.readyThread == 0)
        }
        check(output.all { it == 1000.0 * (index + 1) })
    }
}
