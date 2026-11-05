package com.eignex.koblas

import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.engine.DefaultEngine
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Proxy
import java.net.URLClassLoader
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

class KoblasConfigurationTest {
    @Test
    fun `programmatic override reaches conveniences`() = isolated("invalid") { loader ->
        val facade = loader.loadClass("com.eignex.koblas.Koblas")
        val engineType = loader.loadClass("com.eignex.koblas.KoblasEngine")
        val scalar = loader.loadClass("com.eignex.koblas.engine.BuiltinEngines").getMethod("getScalar").invoke(null)
        var calls = 0
        val supplied = Proxy.newProxyInstance(loader, arrayOf(engineType)) { _, method, args ->
            if (method.name == "sum") calls++
            method.invoke(scalar, *args.orEmpty())
        }
        facade.getMethod("configureKoblas", engineType).invoke(null, supplied)
        val vectorType = loader.loadClass("com.eignex.koblas.Vector")
        val vector = loader.loadClass("com.eignex.koblas.DenseVector")
            .getMethod("of", DoubleArray::class.java).invoke(null, doubleArrayOf(1.0, -2.0, 3.0))

        val result = facade.getMethod("sum", vectorType).invoke(null, vector)

        assertEquals(BuiltinEngines.scalar.sum(doubleArrayOf(1.0, -2.0, 3.0), 0, 3), result)
        assertEquals(1, calls)
        assertSame(supplied, facade.getMethod("getDefault").invoke(null))
    }

    @Test
    fun `JVM launch property selects the scalar engine`() = isolated("scalar") { loader ->
        val scalar = loader.loadClass("com.eignex.koblas.engine.BuiltinEngines").getMethod("getScalar").invoke(null)

        val engine = loader.loadClass("com.eignex.koblas.Koblas").getMethod("getDefault").invoke(null)

        assertSame(scalar, engine)
    }

    @Test
    fun `public configuration rejects changes after first use`() = isolated("scalar") { loader ->
        val facade = loader.loadClass("com.eignex.koblas.Koblas")
        val engineType = loader.loadClass("com.eignex.koblas.KoblasEngine")
        val engine = facade.getMethod("getDefault").invoke(null)

        val failure = assertFailsWith<InvocationTargetException> {
            facade.getMethod("configureKoblas", engineType).invoke(null, engine)
        }

        assertIs<IllegalStateException>(failure.cause)
        assertSame(engine, facade.getMethod("getDefault").invoke(null))
    }

    @Test
    fun `concurrent configuration either wins before selection or is rejected`() {
        val pool = Executors.newFixedThreadPool(2)
        try {
            repeat(20) {
                val automatic = object : KoblasEngine {
                    override val name = "automatic"
                }
                val explicit = BuiltinEngines.scalar
                val selection = DefaultEngine { automatic }
                val start = CountDownLatch(1)
                val configured = pool.submit<Boolean> {
                    start.await()
                    try {
                        selection.configure(explicit)
                        true
                    } catch (_: IllegalStateException) {
                        false
                    }
                }
                val selected = pool.submit<KoblasEngine> {
                    start.await()
                    selection.engine
                }

                start.countDown()
                val won = configured.get(1, TimeUnit.SECONDS)
                val engine = selected.get(1, TimeUnit.SECONDS)

                assertSame(if (won) explicit else automatic, engine)
                assertSame(engine, selection.engine)
            }
        } finally {
            pool.shutdownNow()
            assertTrue(pool.awaitTermination(1, TimeUnit.SECONDS))
        }
    }

    // Each loader owns a fresh Koblas singleton so test order cannot configure another test's engine.
    private fun isolated(setting: String, test: (ClassLoader) -> Unit) {
        val previous = System.getProperty("koblas.engine")
        System.setProperty("koblas.engine", setting)
        try {
            val location = KoblasEngine::class.java.protectionDomain.codeSource.location
            object : URLClassLoader(arrayOf(location), javaClass.classLoader) {
                override fun loadClass(name: String, resolve: Boolean): Class<*> =
                    synchronized(getClassLoadingLock(name)) {
                        val type = if (name.startsWith("com.eignex.koblas.")) {
                            findLoadedClass(name) ?: findClass(name)
                        } else {
                            super.loadClass(name, false)
                        }
                        if (resolve) resolveClass(type)
                        type
                    }
            }.use(test)
        } finally {
            if (previous == null) {
                System.clearProperty("koblas.engine")
            } else {
                System.setProperty("koblas.engine", previous)
            }
        }
    }
}
