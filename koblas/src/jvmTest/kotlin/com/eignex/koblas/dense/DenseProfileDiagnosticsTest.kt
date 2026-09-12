package com.eignex.koblas.dense

import java.net.URLClassLoader
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class DenseProfileDiagnosticsTest {
    @Test
    fun `scalar engine reports malformed shared schedule overrides`() {
        val key = "koblas.dense.packed.block.rows"
        val previous = System.getProperty(key)
        val urls = System.getProperty("java.class.path").split(System.getProperty("path.separator"))
            .map { Path.of(it).toUri().toURL() }.toTypedArray()
        val diagnostics = try {
            System.setProperty(key, "-1")
            URLClassLoader(urls, ClassLoader.getPlatformClassLoader()).use { loader ->
                val engines = loader.loadClass("com.eignex.koblas.BuiltinEngines")
                val scalar = engines.getMethod("getScalar").invoke(null)
                scalar.javaClass.getMethod("getTuningDiagnostics").invoke(scalar).toString()
            }
        } finally {
            if (previous == null) System.clearProperty(key) else System.setProperty(key, previous)
        }

        assertTrue(diagnostics.contains("packed.block.rows=-1"), diagnostics)
        assertTrue(diagnostics.contains("using 128"), diagnostics)
    }
}
