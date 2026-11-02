package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.vendor.*
import kotlin.test.*

class EngineProfileTest {
    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray) =
        assertClose(expected, actual, context = "partial engine composition")

    @Test
    fun `nested compositions retain leaf identity and missing exports reach the next provider`() {
        val vendor = object : KoblasEngine {
            override val name = "host"
            override val vendor = com.eignex.koblas.vendor.Vendor.OneMkl
            override val implementations = setOf(EngineImplementation.Dot)
        }
        val fallback = object : KoblasEngine {
            override val name = "custom"
            override val implementations = setOf(EngineImplementation.Asum)
        }
        val nested = KoblasEngine.combine(EngineProfile("nested", emptyList()), vendor)
        val combined = KoblasEngine.combine(
            EngineProfile(
                "outer",
                listOf(
                    EngineRule(vendor.id, setOf(EngineImplementation.Asum)),
                    EngineRule(fallback.id, fallback.implementations),
                ),
            ),
            nested,
            fallback,
        )

        assertEquals("nested", nested.id)
        assertEquals(null, nested.vendor)
        assertEquals(listOf(vendor, fallback, PortableEngine), combined.providers)
        assertEquals(fallback.name, combined.implementationFor(EngineImplementation.Asum, 128))
    }

    @Test
    fun `threshold and layout select the same body that executes`() {
        val calls = mutableListOf<Int>()
        val backend = object : KoblasEngine {
            override fun requiresUnitStride(implementation: EngineImplementation): Boolean = true
            override fun minimumLength(implementation: EngineImplementation): Int = 2

            override val implementations = setOf(EngineImplementation.Dot)

            override val name: String = "dot"

            @Suppress("LongParameterList")
            override fun dot(
                a: DoubleArray,
                aOff: Int,
                b: DoubleArray,
                bOff: Int,
                len: Int,
                aStride: Int,
                bStride: Int,
            ): Double {
                calls += len
                return PortableEngine.dot(a, aOff, b, bOff, len, aStride, bStride)
            }
        }
        val operation = setOf(EngineImplementation.Dot)
        val rules = listOf(
            EngineRule("dot", operation, rows = 4..Int.MAX_VALUE),
            EngineRule("dot", operation, work = 4),
            EngineRule("dot", operation, columns = 2..Int.MAX_VALUE),
            EngineRule("dot", operation, columns = 0..0),
            EngineRule("dot", operation, depth = 2..Int.MAX_VALUE),
            EngineRule("dot", operation, depth = 0..0),
            EngineRule("dot", operation, borrowed = true),
            EngineRule("dot", operation, contiguous = false),
        )
        val x = DoubleArray(16) { it / 7.0 }
        for (rule in rules) {
            val engine = KoblasEngine.combine(EngineProfile("test", listOf(rule)), backend)
            for (length in listOf(0, 1, 3, 4, 5)) {
                for (stride in listOf(1, 2)) {
                    val expected = PortableEngine.dot(x, 0, x, 1, length, stride, stride)
                    val previous = calls.size
                    val reached = engine.implementationFor(EngineImplementation.Dot, length, stride == 1)
                    val actual = engine.dot(x, 0, x, 1, length, stride, stride)

                    assertEquals(expected, actual)
                    val selected = length >= 2 && stride == 1 && rule.matches(length, 1, 1, true, false)
                    assertEquals(if (selected) "dot" else "scalar", reached)
                    assertEquals(previous + if (reached == "dot") 1 else 0, calls.size)
                }
            }
        }
    }

    @Test
    fun `missing backend and missing capabilities fall through ordered preferences`() {
        val backend = object : KoblasEngine {
            override val name = "sum"
            override val implementations = setOf(EngineImplementation.Sum)
        }
        val engine = KoblasEngine.combine(
            EngineProfile(
                "test",
                listOf(
                    EngineRule("onemkl", setOf(EngineImplementation.Sum)),
                    EngineRule("sum", setOf(EngineImplementation.Sum, EngineImplementation.Dot)),
                ),
            ),
            backend,
        )

        assertEquals("sum", engine.implementationFor(EngineImplementation.Sum, 8))
        assertEquals("scalar", engine.implementationFor(EngineImplementation.Dot, 8))
        assertEquals(
            PortableEngine.nrm2(doubleArrayOf(2.0, 3.0), 0, 2),
            engine.nrm2(doubleArrayOf(2.0, 3.0), 0, 2),
        )
    }

    @Test
    fun `indexed dot registration preserves every window argument`() {
        val counts = mutableListOf<Int>()
        val backend = object : KoblasEngine {

            override val implementations = setOf(EngineImplementation.IndexedDot)
            override val name: String = "indexed"

            @Suppress("LongParameterList")
            override fun dotDense(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                counts += count
                assertEquals(2, indexOffset)
                assertEquals(1, valueOffset)
                return PortableEngine.dotDense(indices, indexOffset, values, valueOffset, count, dense)
            }
        }
        val engine = KoblasEngine.combine(
            EngineProfile(
                "test",
                listOf(
                    EngineRule("indexed", setOf(EngineImplementation.IndexedDot), rows = 2..Int.MAX_VALUE),
                ),
            ),
            backend,
        )
        val indices = intArrayOf(9, 8, 1, 3)
        val values = doubleArrayOf(77.0, 0.5, -0.75)
        val dense = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val expected = BuiltinEngines.scalar.dotDense(indices, 2, values, 1, 2, dense)

        assertEquals(expected, engine.dotDense(indices, 2, values, 1, 2, dense))
        assertEquals(listOf(2), counts)
    }

    @Test
    fun `work boundaries avoid overflow at extreme extents`() {
        val rule = EngineRule("test", setOf(EngineImplementation.Gemm), work = Long.MAX_VALUE)

        assertTrue(rule.matches(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(Int.MAX_VALUE, 1, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(0, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
    }

    @Test
    fun `caller collections cannot change a compiled engine`() {
        val operations = mutableSetOf(EngineImplementation.Sum)
        val rules = mutableListOf(EngineRule("test", operations))
        val profile = EngineProfile("test", rules)
        val backend = object : KoblasEngine {
            override val name = "test"
            override val implementations = setOf(EngineImplementation.Sum)
        }
        val engine = KoblasEngine.combine(profile, backend)
        operations.clear()
        rules.clear()

        assertEquals("test", engine.implementationFor(EngineImplementation.Sum, 8))
        assertEquals(1, profile.rules.size)
    }

    @Test
    fun `conditional product geometry is rejected before packing`() {
        val profile = EngineProfile(
            "test",
            listOf(EngineRule("portable", setOf(EngineImplementation.Products), rows = 32..Int.MAX_VALUE)),
        )

        assertFailsWith<IllegalArgumentException> { KoblasEngine.combine(profile, BuiltinEngines.scalar) }
    }

    @Test
    fun `generated profile parsing preserves bounded routing preferences`() {
        val text = "koblas-profile-v2\ttest\tnative\tLinux\tX86_64\tmeasured cpu\tAvx512\tcapture\tevidence\n" +
            "portable\tDot\t4\t8\t0\t2147483647\t0\t2147483647\t0\ttrue\t*\n"
        val profile = EngineProfile.parse(text)
        val rule = profile.rules.single()

        assertEquals(EngineRuntime.Native, profile.runtime)
        assertEquals("measured cpu", profile.target?.cpuModel)
        assertEquals(CpuClass.Avx512, profile.target?.cpuClass)
        assertTrue(rule.matches(4, 1, 1, true, false))
        assertTrue(rule.matches(8, 1, 1, true, false))
        assertFalse(rule.matches(3, 1, 1, true, false))
        assertFalse(rule.matches(9, 1, 1, true, false))
        assertFalse(rule.matches(4, 1, 1, false, false))
        for ((before, after) in listOf(
            "\tnative\t" to "\tunknown\t",
            "Avx512" to "Unknown",
            "koblas-profile-v2" to "koblas-profile-v1",
            "portable\tDot" to "portable\tUnknown",
            "\t4\t8\t" to "\t8\t4\t",
            "\t4\t8\t" to "\t-1\t8\t",
        )) {
            assertFailsWith<IllegalArgumentException> { EngineProfile.parse(text.replace(before, after)) }
        }
    }

    @Test
    fun `profiles require their measured instruction class`() {
        val profile = EngineProfile(
            "measured",
            emptyList(),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512),
        )

        for (actual in listOf(null, CpuClass.X86_64, CpuClass.Avx2, CpuClass.Avx512)) {
            val host = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = actual)

            assertEquals(actual == CpuClass.Avx512, profile.appliesTo(EngineRuntime.Native, host))
            assertFalse(profile.appliesTo(EngineRuntime.Jvm, host))
        }
        assertEquals(CpuClass.Avx512, cpuClass(Architecture.X86_64, "sse2 avx2 avx512f"))
        assertEquals(CpuClass.Avx2, cpuClass(Architecture.X86_64, "sse2 avx2"))
        assertEquals(CpuClass.X86_64, cpuClass(Architecture.X86_64, "sse2"))
        assertEquals(CpuClass.Arm64, cpuClass(Architecture.Arm64, "fp asimd"))
        assertNull(cpuClass(Architecture.Arm64, null))
        assertEquals(CpuClass.Sve, cpuClass(Architecture.Arm64, "fp asimd sve"))
        assertNull(cpuClass(Architecture.X86_64, null))
    }

    @Test
    fun `partial implementations sharing one backend compose across arithmetic families`() {
        val calls = mutableListOf<String>()
        val vectors = object : KoblasEngine {
            override val name = "vectors"
            override val id = "partial"
            override val implementations = setOf(EngineImplementation.Dot)
            override fun dot(
                a: DoubleArray,
                aOff: Int,
                b: DoubleArray,
                bOff: Int,
                len: Int,
                aStride: Int,
                bStride: Int,
            ): Double {
                calls += "dot"
                return PortableEngine.dot(a, aOff, b, bOff, len, aStride, bStride)
            }
        }
        val indexed = object : KoblasEngine {
            override val name = "indexed"
            override val id = "partial"
            override val implementations = setOf(EngineImplementation.IndexedAxpy)
            override fun axpy(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                alpha: Double,
                destination: DoubleArray,
            ) {
                calls += "indexed update"
                PortableEngine.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
            }
        }
        val matrices = object : KoblasEngine {
            override val name = "matrices"
            override val id = "partial"
            override val implementations = setOf(EngineImplementation.Gemm)
            override fun computeGemm(
                alpha: Double,
                a: DenseMatrixRegion,
                transposeA: Boolean,
                b: DenseMatrixRegion,
                transposeB: Boolean,
                beta: Double,
                c: DenseMatrixRegion,
            ) {
                calls += "matrix product"
                ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
            }
        }
        val operations = vectors.implementations + indexed.implementations + matrices.implementations
        val engine = KoblasEngine.combine(
            EngineProfile("partial", listOf(EngineRule("partial", operations))),
            vectors,
            indexed,
            matrices,
        )
        val nested = KoblasEngine.combine(
            EngineProfile("nested", listOf(EngineRule(engine.id, engine.implementations))),
            engine,
        )
        val values = doubleArrayOf(2.0, 3.0)
        val destination = DoubleArray(2)
        val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expectedProduct = BuiltinEngines.scalar.gemm(matrix, matrix)

        assertEquals(PortableEngine.dot(values, 0, values, 0, 2), nested.dot(values, 0, values, 0, 2))
        engine.axpy(destination, 1.0, intArrayOf(0, 1), 0, values, 0, 2)
        val actualProduct = engine.gemm(matrix, matrix)

        assertContentEquals(values, destination)
        assertAgreesWithReference(expectedProduct.values, actualProduct.values)
        assertEquals("vectors", nested.implementationFor(EngineImplementation.Dot, 2))
        assertEquals(listOf("dot", "indexed update", "matrix product"), calls)
    }
}
