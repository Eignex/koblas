package com.eignex.koblas.internal.backend

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The half of [RunOnce] that needs threads to show: a caller that did not win the gate has to come back with
 * the whole pass behind it, not with however far the winner had got.
 */
class RunOnceBarrierTest {

    @Test
    fun `a second thread waits for the pass rather than returning into a half-done one`() {
        val once = RunOnce()
        val inside = CountDownLatch(1)
        val finished = AtomicBoolean(false)
        val observed = AtomicBoolean(false)

        val owner = thread {
            once.run {
                inside.countDown()
                Thread.sleep(PASS_MILLIS)
                finished.set(true)
            }
        }
        assertTrue(inside.await(WAIT_SECONDS, TimeUnit.SECONDS), "the first caller never entered the pass")
        val waiter = thread {
            once.run { fail("the pass ran a second time") }
            observed.set(finished.get())
        }
        owner.join()
        waiter.join()

        assertTrue(observed.get(), "the second caller returned before the pass had finished")
    }

    private companion object {
        /** Long enough that a caller returning without waiting is caught, short enough for the JVM budget. */
        const val PASS_MILLIS = 100L
        const val WAIT_SECONDS = 5L
    }
}
