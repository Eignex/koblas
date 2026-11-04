package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engineWithHost
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.sparse.*
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Checks how a composition binds rules to providers, then host eligibility, symbol fallback, aliases and
 * argument translation with a recording binding. Small fixtures isolate composition from the platform
 * cutoffs that [StandardEngineProfilesTest] covers.
 */
class CombinedEngineTest {
    @Test
    fun `nested compositions retain leaf identity and missing exports reach the next provider`() {
        val vendor = object : EngineKernels {
            override val name = "host"
            override val vendor = Vendor.OneMkl
            override val implementations = setOf(EngineImplementation.Dot, EngineImplementation.Gemm)
        }
        val fallback = object : EngineKernels {
            override val name = "custom"
            override val implementations = setOf(EngineImplementation.Asum)
        }
        val nested = combine(EngineProfile("nested", emptyList()), vendor)
        val combined = combine(
            EngineProfile(
                "outer",
                listOf(
                    EngineRule(vendor.id, setOf(EngineImplementation.Asum)),
                    EngineRule(vendor.id, setOf(EngineImplementation.Gemm)),
                    EngineRule(fallback.id, fallback.implementations),
                ),
            ),
            nested,
            fallback,
        )

        assertEquals("nested", nested.id)
        assertEquals(null, nested.vendor)
        assertEquals(listOf(vendor, fallback, PortableEngine), combined.providers)
        assertFalse(EngineImplementation.Gemm in combined.implementations)
        assertSame(vendor, combined.wholeCallFor(EngineImplementation.Gemm, 8, 8, 8))
        assertEquals(fallback.name, combined.implementationFor(EngineImplementation.Asum, 128))
    }

    @Test
    fun `threshold and layout select the same body that executes`() {
        val calls = mutableListOf<Int>()
        val backend = object : EngineKernels {
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
            val engine = combine(EngineProfile("test", listOf(rule)), backend)
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
        val backend = object : EngineKernels {
            override val name = "sum"
            override val implementations = setOf(EngineImplementation.Sum)
        }
        val engine = combine(
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
    fun `source typos fail while unavailable built in providers retain coverage`() {
        for (source in listOf("onemlk", "jvm-smid")) {
            val profile = EngineProfile("typo", listOf(EngineRule(source, setOf(EngineImplementation.Dot))))

            assertFailsWith<IllegalArgumentException> { combine(profile) }
        }
        val profile = EngineProfile("optional", listOf(EngineRule("onemkl", setOf(EngineImplementation.Dot))))

        assertEquals(PortableEngine.name, combine(profile).implementationFor(EngineImplementation.Dot, 32))
    }

    @Test
    fun `indexed dot registration preserves every window argument`() {
        val counts = mutableListOf<Int>()
        val backend = object : EngineKernels {

            override val implementations = setOf(EngineImplementation.IndexedDot)
            override val name: String = "indexed"

            @Suppress("LongParameterList")
            override fun dot(
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
                return PortableEngine.dot(indices, indexOffset, values, valueOffset, count, dense)
            }
        }
        val engine = combine(
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
        val expected = BuiltinEngines.scalar.dot(indices, 2, values, 1, 2, dense)

        assertEquals(expected, engine.dot(indices, 2, values, 1, 2, dense))
        assertEquals(listOf(2), counts)
    }

    @Test
    fun `partial implementations sharing one backend compose across arithmetic families`() {
        val calls = mutableListOf<String>()
        val vectors = object : EngineKernels {
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
        val indexed = object : EngineKernels {
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
        val matrices = object : EngineKernels {
            override val name = "matrices"
            override val id = "partial"
            override val implementations = setOf(EngineImplementation.Gemm)
            override fun gemm(
                alpha: Double,
                a: DenseMatrixRegion,
                transposeA: Boolean,
                b: DenseMatrixRegion,
                transposeB: Boolean,
                beta: Double,
                c: DenseMatrixRegion,
                scratch: Scratch?,
            ) {
                calls += "matrix product"
                ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
            }
        }
        val operations = vectors.implementations + indexed.implementations + matrices.implementations
        val engine = combine(
            EngineProfile("partial", listOf(EngineRule("partial", operations))),
            vectors,
            indexed,
            matrices,
        )
        val nested = combine(
            EngineProfile("nested", listOf(EngineRule(engine.id, engine.implementations))),
            engine,
        )
        val values = doubleArrayOf(2.0, 3.0)
        val destination = DoubleArray(2)
        val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expectedProduct = DenseMatrix.zero(2, 2)
        ReferenceBlas.gemm(1.0, matrix, false, matrix, false, 0.0, expectedProduct)
        val actualProduct = DenseMatrix.zero(2, 2)

        assertEquals(PortableEngine.dot(values, 0, values, 0, 2), nested.dot(values, 0, values, 0, 2))
        engine.axpy(intArrayOf(0, 1), 0, values, 0, 2, 1.0, destination)
        engine.gemm(1.0, matrix, false, matrix, false, 0.0, actualProduct)

        assertContentEquals(values, destination)
        assertClose(expectedProduct.values, actualProduct.values, context = "partial engine composition")
        assertEquals("vectors", nested.implementationFor(EngineImplementation.Dot, 2))
        assertEquals(listOf("dot", "indexed update", "matrix product"), calls)
    }

    @Test
    fun `platform eligibility keeps a rejected host call portable`() {
        val recorder = RecordingBlas()
        val portable = engineWithKernels()
        val blas = engineWithHost(portable, recorder, eligible = false)
        val a = matrix(ORDER)
        val actual = matrix(ORDER)
        val expected = matrix(ORDER)
        ReferenceBlas.gemm(0.875, a, false, a, false, -0.25, expected)

        blas.gemm(0.875, a, false, a, false, -0.25, actual)

        assertTrue(recorder.calls.isEmpty())
        assertClose(expected.values, actual.values, context = "host eligibility fallback")
    }

    @Test
    fun `borrowed eligibility rejects any adapted operand before staging`() {
        for (borrowedOperand in listOf("left", "right", "destination")) {
            val recorder = RecordingBlas()
            val portable = engineWithKernels()
            val blas = engineWithHost(portable, recorder, borrowed = false)
            val parent = matrix(ORDER + 2)
            val view = parent.view(1, 1, ORDER, ORDER)
            val a: DenseMatrixRegion = if (borrowedOperand == "left") view else matrix(ORDER)
            val b: DenseMatrixRegion = if (borrowedOperand == "right") view else matrix(ORDER)
            val c: DenseMatrixRegion = if (borrowedOperand == "destination") view else matrix(ORDER)
            val before = c.values.copyOf()
            fun copy(region: DenseMatrixRegion) = DenseMatrix.wrap(
                ORDER,
                ORDER,
                DoubleArray(ORDER * ORDER) { region[it % ORDER, it / ORDER] },
            )
            val expected = copy(c)
            ReferenceBlas.gemm(0.875, copy(a), false, copy(b), false, -0.25, expected)

            blas.gemm(0.875, a, false, b, false, -0.25, c, scratch = Scratch())

            for (j in 0 until ORDER) {
                for (i in 0 until ORDER) {
                    before[c.offset + i + j * c.leadingDimension] = expected[i, j]
                }
            }
            assertTrue(recorder.calls.isEmpty(), borrowedOperand)
            assertClose(before, c.values, context = "$borrowedOperand region eligibility fallback")
        }
    }

    private fun forced(recorder: RecordingBlas = RecordingBlas()): Pair<RecordingBlas, KoblasEngine> =
        recorder to engineWithHost(
            engineWithKernels(),
            recorder,
        )

    private fun matrix(rows: Int, cols: Int = rows, from: Double = 1.0) =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { from + it })

    /** All fourteen bound Level 2 and 3 routines, in the order they are called. */
    @Test
    fun `every level two and three entry point reaches the library when the policy admits it`() {
        val a = matrix(ORDER)
        val b = matrix(ORDER)
        val c = matrix(ORDER)
        val x = DoubleArray(ORDER) { 1.0 + it }
        val y = DoubleArray(ORDER)
        val v = DenseVector.wrap(x)
        val expected = listOf(
            EngineImplementation.Gemv,
            EngineImplementation.Symv,
            EngineImplementation.Ger,
            EngineImplementation.Syr,
            EngineImplementation.Syr2,
            EngineImplementation.Trsv,
            EngineImplementation.Trmv,
            EngineImplementation.Gemm,
            EngineImplementation.Gemmt,
            EngineImplementation.Syrk,
            EngineImplementation.Syr2k, EngineImplementation.Symm, EngineImplementation.Trsm, EngineImplementation.Trmm,
        )
        val (recorder, blas) = forced()

        blas.gemv(1.0, a, x, 0.0, y)
        blas.symv(1.0, a, x, 0.0, y)
        blas.ger(1.0, x, x, matrix(ORDER))
        blas.syr(1.0, v, matrix(ORDER))
        blas.syr2(1.0, v, v, matrix(ORDER))
        blas.trsv(a, x.copyOf(), lower = true)
        blas.trmv(a, x.copyOf(), lower = true)
        blas.gemm(1.0, a, false, b, false, 0.0, c)
        blas.gemmt(1.0, a, false, b, false, 0.0, matrix(ORDER))
        blas.syrk(1.0, a, false, 0.0, matrix(ORDER))
        blas.syr2k(1.0, a, b, false, 0.0, matrix(ORDER))
        blas.symm(1.0, a, b, 0.0, matrix(ORDER))
        blas.trsm(a, matrix(ORDER), lower = true)
        blas.trmm(a, matrix(ORDER), lower = true)

        assertContentEquals(expected, recorder.calls.map { it.operation })
    }

    @Test
    fun `a windowed rank update stays portable even when host policy admits all sizes`() {
        val (recorder, blas) = forced()
        val a = matrix(6)
        val expected = matrix(6)
        val x = DoubleArray(6) { if (it == 0) 0.0 else it.toDouble() }
        ReferenceBlas.ger(-1.0, x, x, expected)

        blas.ger(-1.0, x, x, a, 1, 1, 5, 5, 1, 1)

        assertContentEquals(expected.values, a.values)
        assertEquals(emptyList(), recorder.calls)
    }

    @Test
    fun `a scaled product reaches the library on the same terms as an unscaled one`() {
        val (recorder, blas) = forced()
        val a = matrix(ORDER)

        blas.gemm(SCALE, a, false, a, false, 0.0, matrix(ORDER))

        assertEquals(listOf(EngineImplementation.Gemm), recorder.calls.map { it.operation })
    }

    @Test
    fun `an operation the library does not export stays on the portable schedule`() {
        val without = RecordingBlas(
            implementations =
            EngineImplementation.entries.filter { it.entryPoint.isNotEmpty() }.toSet() - EngineImplementation.Gemmt,
        )
        val (recorder, blas) = forced(without)
        val a = matrix(ORDER)

        blas.gemmt(1.0, a, false, a, false, 0.0, matrix(ORDER))

        assertEquals(emptyList(), recorder.calls.map { it.operation }, "an unexported entry point was called")
    }

    // The engine here admits work of any size, so a zero multiplier stays portable even when every other
    // fact would send the call across.
    @Test
    fun `a call whose contract stops before the arithmetic stays portable at any size`() {
        val (recorder, blas) = forced()
        val a = matrix(ORDER)
        val empty = DenseMatrix.wrap(ORDER, 0, DoubleArray(0))

        blas.gemm(0.0, a, false, a, false, 1.0, matrix(ORDER))
        blas.gemm(1.0, empty, false, DenseMatrix.wrap(0, ORDER, DoubleArray(0)), false, 1.0, matrix(ORDER))

        assertEquals(emptyList(), recorder.calls.map { it.operation }, "a no-work call reached the library")
    }

    @Test
    fun `shared operands reach the library through one snapshot`() {
        for (operation in listOf(
            EngineImplementation.Gemm,
            EngineImplementation.Gemmt,
            EngineImplementation.Symm,
            EngineImplementation.Syr2k,
        )) {
            for (sameMatrix in booleanArrayOf(false, true)) {
                for (useWorkspace in booleanArrayOf(false, true)) {
                    val (recorder, blas) = forced()
                    val c = matrix(ORDER)
                    val b = if (sameMatrix) c else DenseMatrix.wrap(ORDER, ORDER, c.values)
                    val workspace = if (useWorkspace) Scratch() else null

                    when (operation) {
                        EngineImplementation.Gemm -> blas.gemm(
                            0.875,
                            c,
                            false,
                            b,
                            true,
                            -0.25,
                            c,
                            scratch =
                            workspace,
                        )

                        EngineImplementation.Gemmt -> blas.gemmt(
                            0.875,
                            c,
                            false,
                            b,
                            true,
                            -0.25,
                            c,
                            scratch = workspace,
                        )

                        EngineImplementation.Symm -> blas.symm(0.875, c, b, -0.25, c, scratch = workspace)

                        EngineImplementation.Syr2k -> blas.syr2k(0.875, c, b, true, -0.25, c, scratch = workspace)

                        else -> error("unexpected product $operation")
                    }

                    val call = recorder.single()
                    assertEquals(operation, call.operation)
                    assertSame(call.matrices[0].values, call.matrices[1].values, "$operation duplicated its snapshot")
                    assertNotSame(c.values, call.matrices[0].values, "$operation did not stage the shared input")
                    assertContentEquals(c.values, call.matrices[0].values)
                    assertSame(c.values, call.matrices[2].values, "$operation staged its destination")
                    workspace?.let { assertEquals(1, it.available(ORDER * ORDER)) }
                }
            }
        }
    }

    @Test
    fun `host product aliases agree with the reference and share one snapshot`() {
        val portable = engineWithKernels()
        val blas = engineWithHost(portable, ReferenceProductBlas())

        assertDenseProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `host rectangular product aliases retain their operand shapes`() {
        val portable = engineWithKernels()
        val blas = engineWithHost(portable, ReferenceProductBlas())

        assertRectangularProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `the staged copy is borrowed from the caller s workspace`() {
        val (_, blas) = forced()
        val workspace = Scratch()
        val shared = DoubleArray(ORDER * ORDER) { 1.0 + it }
        val c = DenseMatrix.wrap(ORDER, ORDER, shared)

        blas.gemm(
            1.0,
            DenseMatrix.wrap(ORDER, ORDER, shared),
            false,
            matrix(ORDER),
            false,
            0.0,
            c,
            scratch =
            workspace,
        )
        val lent = workspace.available(ORDER * ORDER)
        blas.gemm(
            1.0,
            DenseMatrix.wrap(ORDER, ORDER, shared),
            false,
            matrix(ORDER),
            false,
            0.0,
            c,
            scratch =
            workspace,
        )

        assertEquals(1, lent, "the staging was not lent by the workspace")
        assertEquals(1, workspace.available(ORDER * ORDER), "a second call allocated its own staging")
    }

    @Test
    fun `borrowed host products pack logical entries`() {
        val (recorder, blas) = forced()
        val parent = matrix(7)
        val a = parent.view(1, 1, 3, 2)
        val b = parent.view(0, 3, 2, 3)
        val c = parent.view(2, 2, 3, 3)
        val expectedA = a.toArray()
        val expectedB = b.toArray()

        blas.gemm(1.0, a, false, b, false, 0.0, c, scratch = Scratch())

        val call = recorder.single()
        assertEquals(EngineImplementation.Gemm, call.operation)
        for (i in expectedA.indices) assertContentEquals(expectedA[i], call.matrices[0].toArray()[i])
        for (i in expectedB.indices) assertContentEquals(expectedB[i], call.matrices[1].toArray()[i])
        assertNotSame(parent.values, call.matrices[0].values)
    }

    @Test
    fun `borrowed host calls reach a region binding in place`() {
        val recorder = RecordingBlas()
        val blas = engineWithHost(
            engineWithKernels(),
            recorder,
        )
        val parent = matrix(9)
        val panel = parent.view(4, 0, 5, 3)
        val trailing = parent.view(4, 4, 5, 5)
        val triangle = parent.view(0, 0, 3, 3)
        val beside = parent.view(0, 4, 3, 5)
        val workspace = Scratch()

        blas.gemv(1.0, panel, DoubleArray(3), 0.0, DoubleArray(5), scratch = workspace)
        blas.gemm(1.0, panel, false, beside, false, 1.0, trailing, scratch = workspace)
        blas.syrk(-1.0, panel, false, 1.0, trailing, scratch = workspace)
        blas.trsv(triangle, DoubleArray(3), lower = true, scratch = workspace)
        blas.trsm(triangle, beside, lower = true, scratch = workspace)

        val expected = listOf(
            listOf(panel),
            listOf(panel, beside, trailing),
            listOf(panel, trailing),
            listOf(triangle),
            listOf(triangle, beside),
        )
        assertEquals(expected.size, recorder.calls.size)
        for ((call, operands) in recorder.calls.zip(expected)) {
            for ((actual, view) in call.matrices.zip(operands)) assertSame(view, actual, "${call.operation}")
        }
        assertEquals(0, workspace.idleLengths(), "a disjoint borrowed call borrowed scratch")
    }

    @Test
    fun `a borrowed input overlapping the destination is staged before a region binding`() {
        val recorder = RecordingBlas()
        val blas = engineWithHost(
            engineWithKernels(),
            recorder,
        )
        val parent = matrix(7)
        val a = parent.view(0, 0, 4, 3)
        val c = parent.view(1, 1, 4, 4)
        val entries = a.toArray()

        blas.syrk(-1.0, a, false, 1.0, c, scratch = Scratch())

        val call = recorder.calls.single()
        assertNotSame(parent.values, call.matrices[0].values)
        for (i in entries.indices) assertContentEquals(entries[i], call.matrices[0].toArray()[i])
        assertSame(c, call.matrices[1])
    }

    private companion object {
        /** Small enough that only a forced policy sends it over, which is what these cases want. */
        const val ORDER = 3

        const val SCALE = 0.75
    }
}

/** Computes host products from the received operands, so staging is checked without an installed library. */
@Suppress("LongParameterList") // the host BLAS product signatures
private class ReferenceProductBlas : RecordingBlas() {
    override fun cblasGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ) = ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c)

    override fun cblasGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = ReferenceBlas.gemmt(alpha, a, transposeA, b, transposeB, beta, c, structure == MatrixStructure.SymmetricLower)

    override fun cblasSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ) = ReferenceBlas.symm(alpha, a, b, beta, c, structure == MatrixStructure.SymmetricLower, rightSide)

    override fun cblasSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = ReferenceBlas.syr2k(alpha, a, b, transposeA, beta, c, structure == MatrixStructure.SymmetricLower)
}
