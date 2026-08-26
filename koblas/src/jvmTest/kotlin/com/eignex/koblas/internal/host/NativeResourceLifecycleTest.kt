package com.eignex.koblas.internal.host

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

class NativeResourceLifecycleTest {
    @Test
    fun `close waits for an active call and releases once`() {
        var releases = 0
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val closeStarted = CountDownLatch(1)
        val releaseStarted = CountDownLatch(1)
        val lifecycle = NativeResourceLifecycle("test resource") {
            releases++
            releaseStarted.countDown()
        }
        val executor = Executors.newFixedThreadPool(2)

        try {
            val call = executor.submit {
                lifecycle.withResource {
                    entered.countDown()
                    check(finish.await(1, TimeUnit.SECONDS))
                }
            }
            check(entered.await(1, TimeUnit.SECONDS))
            val close = executor.submit {
                closeStarted.countDown()
                lifecycle.run()
            }
            check(closeStarted.await(1, TimeUnit.SECONDS))
            assertFalse(releaseStarted.await(50, TimeUnit.MILLISECONDS), "resource released during an active call")
            finish.countDown()
            call.get(1, TimeUnit.SECONDS)
            close.get(1, TimeUnit.SECONDS)
            lifecycle.run()

            assertEquals(1, releases)
            assertFailsWith<IllegalStateException> { lifecycle.withResource {} }
        } finally {
            executor.shutdownNow()
        }
    }
}
