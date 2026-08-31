package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals

class RegistrationConcurrencyTest {
    private class Ranked(override val name: String, override val priority: Int) :
        F64Blas by F64ReferenceLinearAlgebra {
        override val isAvailable: Boolean get() = true
        override val isPortable: Boolean get() = false
    }

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
                        registerBackend(Ranked("concurrent-$priority", priority))
                    }
                }.forEach { it.get() }
                assertEquals(priorities.max(), koblas.blas.priority)
            }
        } finally {
            pool.shutdownNow()
        }
    }
}
