package com.eignex.koblas

import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals

class RegistrationConcurrencyTest {
    @Test
    fun `concurrent registrations keep the strongest offer`() = withCleanBackends {
        val threads = 4
        val pool = Executors.newFixedThreadPool(threads)
        try {
            repeat(40) { round ->
                val priorities = (1..threads).map { round * 10 + it }
                val barrier = CyclicBarrier(threads)
                priorities.map { priority ->
                    pool.submit {
                        barrier.await()
                        registerBackend(RankedBlas("concurrent-$priority", priority))
                    }
                }.forEach { it.get() }
                assertEquals(priorities.max(), koblas.blas.priority)
            }
        } finally {
            pool.shutdownNow()
        }
    }
}
