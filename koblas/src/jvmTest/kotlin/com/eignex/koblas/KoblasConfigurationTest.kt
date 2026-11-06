package com.eignex.koblas

import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Proxy
import java.net.URLClassLoader
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertSame

class KoblasConfigurationTest {
    @Test
    fun `programmatic override reaches conveniences`() = isolated("scalar") { loader ->
        val facade = loader.loadClass("com.eignex.koblas.Koblas")
        val engineType = loader.loadClass("com.eignex.koblas.KoblasEngine")
        val scalar = loader.loadClass("com.eignex.koblas.KoblasEngine").getMethod("getScalar").invoke(null)
        var calls = 0
        val supplied = Proxy.newProxyInstance(loader, arrayOf(engineType)) { _, method, args ->
            if (method.name == "sum") calls++
            method.invoke(scalar, *args.orEmpty())
        }
        facade.getMethod("setDefault", engineType).invoke(null, supplied)
        val vectorType = loader.loadClass("com.eignex.koblas.Vector")
        val vector = loader.loadClass("com.eignex.koblas.DenseVector")
            .getMethod("of", DoubleArray::class.java).invoke(null, doubleArrayOf(1.0, -2.0, 3.0))

        val result = facade.getMethod("sum", vectorType).invoke(null, vector)

        assertEquals(KoblasEngine.scalar.sum(DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))), result)
        assertEquals(1, calls)
        assertSame(supplied, facade.getMethod("getDefault").invoke(null))
    }

    @Test
    fun `JVM launch property selects the scalar engine`() = isolated("scalar") { loader ->
        val scalar = loader.loadClass("com.eignex.koblas.KoblasEngine").getMethod("getScalar").invoke(null)

        val engine = loader.loadClass("com.eignex.koblas.Koblas").getMethod("getDefault").invoke(null)

        assertSame(scalar, engine)
    }

    @Test
    fun `invalid launch settings fail before a programmatic override`() = isolated("invalid") { loader ->
        val facade = loader.loadClass("com.eignex.koblas.Koblas")
        val engineType = loader.loadClass("com.eignex.koblas.KoblasEngine")
        val scalar = engineType.getMethod("getScalar").invoke(null)

        val failure = assertFailsWith<InvocationTargetException> {
            facade.getMethod("setDefault", engineType).invoke(null, scalar)
        }

        val initialization = assertIs<ExceptionInInitializerError>(failure.cause)
        assertIs<IllegalArgumentException>(initialization.cause)
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
