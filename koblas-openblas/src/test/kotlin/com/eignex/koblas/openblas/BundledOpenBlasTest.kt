package com.eignex.koblas.openblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BundledOpenBlasTest {
    @Test
    fun `registers through the backend service`() {
        assertEquals("openblas-bundled", koblas.blas.name)
        assertEquals("openblas-bundled", koblas.lapack.name)
    }

    @Test
    fun `loads the Maven OpenBLAS binary`() {
        val backend = BundledOpenBlas()

        assertTrue(backend.isAvailable)
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 4.0)))
        val out = F64DenseMatrix.zero(2)
        backend.gemm(1.0, a, false, a, false, 0.0, out)

        assertEquals(listOf(7.0, 10.0, 15.0, 22.0), out.data.toList())
    }

    @Test
    fun `uses a private extraction directory`() {
        val path = OpenBlasResources.extract().openblas

        assertTrue(path.parent.fileName.toString().startsWith("koblas-openblas-"))
        assertTrue(Files.isDirectory(path.parent))
    }

    @Test
    fun `preserves explicit host library paths`() {
        val openblas = Files.createTempFile("openblas-user-", ".so")
        val lapacke = Files.createTempFile("lapacke-user-", ".so")
        val bundledOpenBlas = Files.createTempFile("openblas-bundled-", ".so")
        val bundledLapacke = Files.createTempFile("lapacke-bundled-", ".so")
        val oldOpenBlas = System.getProperty("koblas.openblas.path")
        val oldLapacke = System.getProperty("koblas.lapacke.path")
        try {
            System.setProperty("koblas.openblas.path", openblas.toString())
            System.setProperty("koblas.lapacke.path", lapacke.toString())

            configureHostLibraryPaths(OpenBlasPaths(bundledOpenBlas, bundledLapacke))

            assertEquals(openblas.toString(), System.getProperty("koblas.openblas.path"))
            assertEquals(lapacke.toString(), System.getProperty("koblas.lapacke.path"))
        } finally {
            restoreProperty("koblas.openblas.path", oldOpenBlas)
            restoreProperty("koblas.lapacke.path", oldLapacke)
            Files.deleteIfExists(openblas)
            Files.deleteIfExists(lapacke)
            Files.deleteIfExists(bundledOpenBlas)
            Files.deleteIfExists(bundledLapacke)
        }
    }

    private fun restoreProperty(name: String, value: String?) {
        if (value == null) System.clearProperty(name) else System.setProperty(name, value)
    }
}
