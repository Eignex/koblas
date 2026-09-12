package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import java.lang.management.ManagementFactory
import kotlin.test.Test
import kotlin.test.assertTrue

class SparseSlicesAllocationTest {
    @Test
    fun `reuse comparisons allocate no buffers while running`() {
        val bean = ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!bean.isThreadAllocatedMemorySupported) return
        bean.isThreadAllocatedMemoryEnabled = true
        val works = sparseSlicesComparisonOperations.map { operation ->
            val case = Cases.parse("$operation+64+sparse-uniform+density=0.125+timing=reuse+locality=shuffled").single()
            SparseSlicesReuseWork(case, BuiltinEngines.scalar)
        }.toTypedArray()
        repeat(2_000) { for (work in works) sink = work.run() }
        @Suppress("DEPRECATION")
        val thread = Thread.currentThread().id

        val before = bean.getThreadAllocatedBytes(thread)
        repeat(2_000) { for (work in works) sink = work.run() }
        val allocated = bean.getThreadAllocatedBytes(thread) - before

        assertTrue(allocated.toDouble() / (2_000 * works.size) <= 1.0, "allocated $allocated bytes across reusable calls")
    }

    companion object {
        @Volatile private var sink = 0.0
    }
}
