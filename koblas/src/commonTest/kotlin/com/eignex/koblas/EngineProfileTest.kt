package com.eignex.koblas

import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.dense.implementationFor
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.vendor.Architecture
import com.eignex.koblas.vendor.CpuVendor
import com.eignex.koblas.vendor.HostPlatform
import com.eignex.koblas.vendor.OperatingSystem
import kotlin.test.*

class EngineProfileTest {
    @Test
    fun `threshold and layout select the same body that executes`() {
        val calls = mutableListOf<Int>()
        val backend = EngineBackend(
            "dot",
            vectors = object : DenseVectorKernels {
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
                    return ScalarVectorKernels.dot(a, aOff, b, bOff, len, aStride, bStride)
                }
            },
            implementations = setOf(EngineImplementation.Dot),
            contiguousVectors = true,
            minimumVectorLength = 2,
        )
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
                    val expected = ScalarVectorKernels.dot(x, 0, x, 1, length, stride, stride)
                    val previous = calls.size
                    val reached = engine.vectorKernels.implementationFor(EngineImplementation.Dot, length, stride == 1)
                    val actual = engine.vectorKernels.dot(x, 0, x, 1, length, stride, stride)

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
        val backend = EngineBackend("sum", implementations = setOf(EngineImplementation.Sum))
        val engine = KoblasEngine.combine(
            EngineProfile(
                "test",
                listOf(
                    EngineRule("absent", setOf(EngineImplementation.Sum)),
                    EngineRule("sum", setOf(EngineImplementation.Sum, EngineImplementation.Dot)),
                ),
            ),
            backend,
        )

        assertEquals("sum", engine.vectorKernels.implementationFor(EngineImplementation.Sum, 8))
        assertEquals("scalar", engine.vectorKernels.implementationFor(EngineImplementation.Dot, 8))
        assertEquals(
            ScalarVectorKernels.nrm2(doubleArrayOf(2.0, 3.0), 0, 2),
            engine.vectorKernels.nrm2(doubleArrayOf(2.0, 3.0), 0, 2),
        )
    }

    @Test
    fun `indexed dot registration preserves every window argument`() {
        val counts = mutableListOf<Int>()
        val backend = EngineBackend(
            "indexed",
            indexedDot = object : IndexedSparseKernels {
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
                    return ScalarIndexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
                }
            },
        )
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
        val expected = BuiltinEngines.scalar.indexedSparseKernels.dotDense(indices, 2, values, 1, 2, dense)

        assertEquals(expected, engine.indexedSparseKernels.dotDense(indices, 2, values, 1, 2, dense))
        assertEquals(listOf(2), counts)
    }

    @Test
    fun `profile applicability distinguishes runtime architecture and manufacturer`() {
        val profile = StandardEngineProfiles.nativeLinuxOneMkl
        val host = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Intel)

        assertTrue(profile.appliesTo("native", host))
        assertFalse(profile.appliesTo("jvm", host))
        assertFalse(
            profile.appliesTo("native", HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Amd)),
        )
        assertFalse(
            profile.appliesTo("native", HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, CpuVendor.Unknown)),
        )
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
        val backend = EngineBackend("test", implementations = setOf(EngineImplementation.Sum))
        val engine = KoblasEngine.combine(profile, backend)
        operations.clear()
        rules.clear()

        assertEquals("test", engine.vectorKernels.implementationFor(EngineImplementation.Sum, 8))
        assertEquals(1, profile.rules.size)
    }

    @Test
    fun `conditional product geometry is rejected before packing`() {
        val profile = EngineProfile(
            "test",
            listOf(EngineRule("portable", setOf(EngineImplementation.Products), rows = 32..Int.MAX_VALUE)),
        )

        assertFailsWith<IllegalArgumentException> { KoblasEngine.combine(profile, EngineBackends.portable) }
    }

    @Test
    fun `standard host rules preserve measured layout and shape boundaries`() {
        fun eligible(
            profile: EngineProfile,
            operation: EngineImplementation,
            rows: Int,
            columns: Int,
            depth: Int,
            borrowed: Boolean = false,
        ): Boolean = profile.rules.any {
            operation in it.implementations && it.matches(rows, columns, depth, true, borrowed)
        }
        val native = StandardEngineProfiles.nativeLinuxOneMkl
        assertTrue(eligible(native, EngineImplementation.Gemv, 32, 32, 1))
        assertFalse(eligible(native, EngineImplementation.Gemv, 32, 31, 1))
        assertTrue(eligible(native, EngineImplementation.Syrk, 8, 8, 32, borrowed = true))
        assertFalse(eligible(native, EngineImplementation.Gemm, 128, 128, 0))
        val jvm = StandardEngineProfiles.jvmLinuxOneMkl
        for (op in listOf(
            EngineImplementation.Gemm,
            EngineImplementation.Symm,
            EngineImplementation.Syrk,
            EngineImplementation.Syr2k,
            EngineImplementation.Trmm,
            EngineImplementation.Trsm,
        )) {
            assertTrue(eligible(jvm, op, 64, 64, 64))
            assertFalse(eligible(jvm, op, 63, 64, 64))
            assertFalse(eligible(jvm, op, 64, 63, 64))
            assertFalse(eligible(jvm, op, 64, 64, 63))
            assertFalse(eligible(jvm, op, 127, 127, 127, borrowed = true))
            assertTrue(eligible(jvm, op, 128, 128, 128, borrowed = true))
        }
        assertFalse(eligible(jvm, EngineImplementation.Gemmt, 64, 64, 64))
        assertTrue(eligible(jvm, EngineImplementation.Gemmt, 128, 128, 128))
        assertTrue(eligible(jvm, EngineImplementation.Gemm, 16, 16, 1024))
        assertTrue(eligible(jvm, EngineImplementation.Gemm, 16, 16, 4096))
        assertFalse(eligible(jvm, EngineImplementation.Gemm, 16, 16, 4097))
        assertFalse(eligible(jvm, EngineImplementation.Gemm, 16, 16, 1024, borrowed = true))
    }

    @Test
    fun `generated profile parsing preserves bounded routing preferences`() {
        val text = "koblas-profile-v1\ttest\tnative\tLinux\tX86_64\tIntel\tcapture\tevidence\n" +
            "portable\tDot\t4\t8\t0\t2147483647\t0\t2147483647\t0\ttrue\t*\n"
        val profile = EngineProfile.parse(text)
        val rule = profile.rules.single()

        assertTrue(rule.matches(4, 1, 1, true, false))
        assertTrue(rule.matches(8, 1, 1, true, false))
        assertFalse(rule.matches(3, 1, 1, true, false))
        assertFalse(rule.matches(9, 1, 1, true, false))
        assertFalse(rule.matches(4, 1, 1, false, false))
        for ((before, after) in listOf(
            "portable\tDot" to "portable\tUnknown",
            "\t4\t8\t" to "\t8\t4\t",
            "\t4\t8\t" to "\t-1\t8\t",
        )) {
            assertFailsWith<IllegalArgumentException> { EngineProfile.parse(text.replace(before, after)) }
        }
    }
}
