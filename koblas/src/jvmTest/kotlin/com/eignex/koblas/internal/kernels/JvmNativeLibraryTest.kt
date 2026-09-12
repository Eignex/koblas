package com.eignex.koblas.internal.kernels

import java.net.URL
import java.net.URLClassLoader
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class JvmNativeLibraryTest {
    @Test
    fun `missing native resources leave the scalar engine available`() {
        checkUnavailable(null)
    }

    @Test
    fun `old native libraries leave the scalar engine available`() {
        checkUnavailable("old")
    }

    @Test
    fun `incomplete native libraries leave the scalar engine available`() {
        checkUnavailable("incomplete")
    }

    @Test
    fun `incompatible probe versions leave the scalar engine available`() {
        checkUnavailable("incompatible")
    }

    private fun checkUnavailable(fixture: String?) {
        val fixtureUrl = fixture?.let {
            Path.of(checkNotNull(System.getProperty("koblas.test.nativeFixtures")), "$it.so").toUri().toURL()
        }
        val urls = System.getProperty("java.class.path").split(System.getProperty("path.separator"))
            .map { Path.of(it).toUri().toURL() }.toTypedArray()
        val loader = object : URLClassLoader(urls, ClassLoader.getPlatformClassLoader()) {
            override fun getResource(name: String): URL? =
                if (name.startsWith("com/eignex/koblas/internal/kernels/") && name.contains("libkoblas")) {
                    fixtureUrl
                } else {
                    super.getResource(name)
                }
        }
        val (native, name) = loader.use {
            val thread = Thread.currentThread()
            val previous = thread.contextClassLoader
            thread.contextClassLoader = loader
            try {
                val enginesClass = loader.loadClass("com.eignex.koblas.BuiltinEngines")
                val engines = enginesClass.getField("INSTANCE").get(null)
                val native = enginesClass.getMethod("getC").invoke(engines)
                val scalar = checkNotNull(enginesClass.getMethod("getScalar").invoke(engines))
                val vectors = scalar.javaClass.getMethod("getVectorKernels").invoke(scalar)
                val vectorClass = loader.loadClass("com.eignex.koblas.dense.DenseVectorKernels")
                native to vectorClass.getMethod("getName").invoke(vectors)
            } finally {
                thread.contextClassLoader = previous
            }
        }
        assertNull(native)
        assertEquals("scalar", name)
    }
}
