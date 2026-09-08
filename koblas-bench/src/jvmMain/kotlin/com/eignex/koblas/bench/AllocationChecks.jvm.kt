package com.eignex.koblas.bench

import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

internal actual fun verifyNearZeroManagedAllocation(label: String, operation: () -> Unit) {
    val bean = ManagementFactory.getThreadMXBean() as? ThreadMXBean ?: return
    if (!bean.isThreadAllocatedMemorySupported) return
    bean.isThreadAllocatedMemoryEnabled = true
    repeat(20_000) { operation() }
    val thread = Thread.currentThread().threadId()
    val before = bean.getThreadAllocatedBytes(thread)
    repeat(10_000) { operation() }
    val bytesPerCall = (bean.getThreadAllocatedBytes(thread) - before) / 10_000
    check(bytesPerCall <= 64) {
        "$label allocated $bytesPerCall B per call; this fork may have lost vector intrinsification"
    }
    println("allocation: operation=$label expectation=near-zero measured=$bytesPerCall B/call")
}
