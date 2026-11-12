package com.eignex.koblas

import com.eignex.koblas.internal.operations.*
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Proxy
import java.net.URLClassLoader
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame

class KoblasConfigurationTest {
    @Test
    fun `programmatic override reaches every typed dot overload`() = recording { engine ->
        for (a in vectors()) {
            for (b in vectors()) {
                val result = engine.call("dot", arrayOf(Vector::class.java, Vector::class.java), a, b)

                assertEquals(KoblasEngine.scalar.dotVector(a, b), result)
                engine.assertCall("dot", a.javaClass, b.javaClass)
            }
        }
    }

    @Test
    fun `programmatic override reaches vector reductions`() = recording { engine ->
        for (vector in vectors()) {
            for (type in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) {
                val result = engine.call("norm", arrayOf(Vector::class.java, Norm::class.java), vector, type)

                assertEquals(KoblasEngine.scalar.normVector(vector, type), result)
                val operation = when (type) {
                    Norm.One -> "asum"
                    Norm.Two -> "nrm2"
                    else -> "iamax"
                }
                engine.assertCall(operation, if (operation == "iamax") DenseVector::class.java else vector.javaClass)
            }
            for (operation in listOf("sum", "asum")) {
                engine.call(operation, arrayOf(Vector::class.java), vector)

                engine.assertCall(operation, vector.javaClass)
            }
        }
    }

    @Test
    fun `programmatic override reaches matrix reductions`() = recording { engine ->
        for (matrix in matrices()) {
            val sum = engine.call("sum", arrayOf(Matrix::class.java), matrix)

            assertEquals(KoblasEngine.scalar.sumMatrix(matrix), sum)
            engine.assertCalls("sum")
            for (axis in 0..1) {
                val result = engine.call(
                    "sum",
                    arrayOf(Matrix::class.java, Int::class.javaPrimitiveType!!),
                    matrix,
                    axis,
                )

                assertContentEquals(KoblasEngine.scalar.sumMatrix(matrix, axis), (result as Vector).toDoubleArray())
                engine.assertCalls("sum", required = matrix !is SparseMatrix || axis == 0)
            }
            for (type in listOf(Norm.Frobenius, Norm.One, Norm.Infinity, Norm.MaxAbs)) {
                val result = engine.call("norm", arrayOf(Matrix::class.java, Norm::class.java), matrix, type)

                assertEquals(KoblasEngine.scalar.normMatrix(matrix, type), result)
                val operation = when (type) {
                    Norm.Frobenius -> "nrm2"
                    Norm.MaxAbs -> "iamax"
                    else -> "asum"
                }
                engine.assertCalls(operation, required = matrix !is SparseMatrix || type != Norm.Infinity)
            }
        }
    }

    @Test
    fun `programmatic override reaches every typed copy overload`() = recording { engine ->
        for (source in vectors()) {
            for (destination in vectors()) {
                val expected = destination.copy()
                KoblasEngine.scalar.copyIntoVector(source, expected)

                engine.call("copyFrom", arrayOf(Vector::class.java, Vector::class.java), destination, source)

                assertContentEquals(expected.toDoubleArray(), destination.toDoubleArray())
                engine.assertCall("copy", source.javaClass, destination.javaClass)
            }
        }
    }

    @Test
    fun `programmatic override reaches staged opaque vector copies`() = recording { engine ->
        for (columnSource in listOf(false, true)) {
            val backing = doubleArrayOf(1.0, -2.0, 3.0, 4.0, 5.0, 6.0)
            val source = DenseVector.wrap(backing, 0, 3)
            val opaque = if (columnSource) {
                val matrix = object : Matrix by DenseMatrix.wrap(3, 2, backing) {}
                matrix.columnView(0)
            } else {
                object : Vector by source {}
            }
            val destination = DenseVector.wrap(backing, 2, 3)
            val expected = backing.copyOf()
            KoblasEngine.scalar.copy(DenseVector.wrap(expected, 0, 3), DenseVector.wrap(expected, 2, 3))

            engine.call("copyFrom", arrayOf(Vector::class.java, Vector::class.java), destination, opaque)

            assertContentEquals(expected, backing)
            engine.assertCall("copy", DenseVector::class.java, DenseVector::class.java)
        }
    }

    @Test
    fun `programmatic override reaches matrix copying with aliases`() = recording { engine ->
        for (source in matrices()) {
            for (destination in matrices()) {
                val expected = destination.copy()
                copyIntoMatrix(source, expected)

                engine.call("copyFrom", arrayOf(Matrix::class.java, Matrix::class.java), destination, source)

                for (j in 0 until source.cols) {
                    for (i in 0 until source.rows) assertEquals(expected[i, j], destination[i, j])
                }
                engine.assertCall("copy", source.javaClass, destination.javaClass)
            }
        }
        val backing = DoubleArray(12) { it.toDouble() }
        val parent = DenseMatrix.wrap(3, 4, backing)
        val source = parent.view(0, 0, 2, 3)
        val destination = parent.view(1, 1, 2, 3)
        val expected = backing.copyOf()
        val reference = DenseMatrix.wrap(3, 4, expected)
        copyIntoMatrix(reference.view(0, 0, 2, 3), reference.view(1, 1, 2, 3))

        engine.call("copyFrom", arrayOf(Matrix::class.java, Matrix::class.java), destination, source)

        assertContentEquals(expected, backing)
        engine.assertCalls("copy")
    }

    @Test
    fun `launch and programmatic selections have equivalent coverage`() {
        val names = listOf("scalar", "simd", "auto", "invalid", " ") + Vendor.entries.map { it.vendorName }
        for (name in names) {
            isolated(name) { loader ->
                val selection = runCatching { KoblasEngine.select(name) }
                val launch = runCatching { defaultEngine(loader) }
                if (selection.isFailure) {
                    val launchFailure = assertIs<InvocationTargetException>(launch.exceptionOrNull()).cause
                    val initialization = assertIs<ExceptionInInitializerError>(launchFailure)
                    assertEquals(selection.exceptionOrNull()?.javaClass, initialization.cause?.javaClass)
                    assertEquals(selection.exceptionOrNull()?.message, initialization.cause?.message)
                    Vendor.entries.firstOrNull { it.vendorName == name }?.let { vendor ->
                        assertNull(KoblasEngine.vendor(vendor))
                        println("SKIPPED: ${vendor.vendorName} unavailable; fake providers cover vendor fallback")
                    }
                } else {
                    val selected = selection.getOrThrow()
                    assertEquivalent(selected, launch.getOrThrow())
                    Vendor.entries.firstOrNull { it.vendorName == name }?.let { vendor ->
                        assertEquivalent(selected, requireNotNull(KoblasEngine.vendor(vendor)))
                    }
                }
            }
        }
    }

    @Test
    fun `environment selection uses the same programmatic resolver`() = isolated(null) { loader ->
        val selected = KoblasEngine.select(System.getenv("KOBLAS_ENGINE") ?: "auto")

        assertEquivalent(selected, defaultEngine(loader))
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

    private fun defaultEngine(loader: ClassLoader): KoblasEngine = loader.loadClass("com.eignex.koblas.Koblas")
        .getMethod("getDefault").invoke(null) as KoblasEngine

    private fun assertEquivalent(selected: KoblasEngine, configured: KoblasEngine) {
        assertEquals(selected.name, configured.name)
        assertEquals(selected.providers.map { it.id }, configured.providers.map { it.id })
        val vector = DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        assertEquals(KoblasEngine.scalar.sum(vector), selected.sum(vector))
        assertEquals(selected.sum(vector), configured.sum(vector))
        assertEquals(selected.dot(vector, vector), configured.dot(vector, vector))
    }

    private fun vectors(): List<Vector> = listOf(
        DenseVector.wrap(doubleArrayOf(99.0, 3.0, 99.0, -2.0, 99.0, 1.0), 5, 3, -2),
        SparseVector.wrap(3, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0)),
    )

    private fun matrices(): List<Matrix> = listOf(
        DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, -2.0, 3.0, 4.0)),
        DenseMatrix.wrap(3, 3, DoubleArray(9) { it.toDouble() }).view(1, 1, 2, 2),
        SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 3.0, 1 to -4.0))),
    )

    private fun recording(test: (RecordingEngine) -> Unit) = isolated("scalar") { loader ->
        val facade = loader.loadClass("com.eignex.koblas.Koblas")
        val calls = mutableListOf<Pair<String, List<Class<*>>>>()
        val scalar = KoblasEngine.scalar
        val supplied = Proxy.newProxyInstance(loader, arrayOf(KoblasEngine::class.java)) { _, method, args ->
            calls += method.name to method.parameterTypes.toList()
            method.invoke(scalar, *args.orEmpty())
        }
        assertFalse(supplied is com.eignex.koblas.internal.kernels.KernelProvider)
        facade.getMethod("setDefault", KoblasEngine::class.java).invoke(null, supplied)

        test(RecordingEngine(facade, calls))

        assertSame(supplied, facade.getMethod("getDefault").invoke(null))
    }

    private class RecordingEngine(val facade: Class<*>, val calls: MutableList<Pair<String, List<Class<*>>>>) {
        fun call(name: String, types: Array<Class<*>>, vararg args: Any): Any? =
            facade.getMethod(name, *types).invoke(null, *args)

        fun assertCall(name: String, vararg types: Class<*>) {
            assertEquals(listOf(name to types.toList()), calls)
            calls.clear()
        }

        fun assertCalls(name: String, required: Boolean = true) {
            assertEquals(required, calls.isNotEmpty())
            for (call in calls) assertEquals(name, call.first)
            calls.clear()
        }
    }

    // Each loader owns a fresh Koblas singleton so test order cannot configure another test's engine.
    private fun isolated(setting: String?, test: (ClassLoader) -> Unit) {
        val previous = System.getProperty("koblas.engine")
        if (setting == null) System.clearProperty("koblas.engine") else System.setProperty("koblas.engine", setting)
        try {
            val location = KoblasEngine::class.java.protectionDomain.codeSource.location
            object : URLClassLoader(arrayOf(location), javaClass.classLoader) {
                override fun loadClass(name: String, resolve: Boolean): Class<*> =
                    synchronized(getClassLoadingLock(name)) {
                        val facade = name == "com.eignex.koblas.Koblas" ||
                            name.startsWith("com.eignex.koblas.Koblas__")
                        val type = if (facade) {
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
