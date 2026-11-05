package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operations.MatrixStructure
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** Single-host dispatch, missing exports, alias staging and argument translation. */
class BuiltinEngineTest {
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
            val order = 8
            val blas = BuiltinEngine(
                "test",
                host = recorder,
                tuning = EngineTuning(EngineRuntime.Native, HostPlatform(OperatingSystem.Linux, Architecture.X86_64)),
            )
            assertSame(recorder, blas.wholeCallFor(EngineImplementation.Gemm, order, order, order))
            val parent = matrix(order + 2)
            val view = parent.view(1, 1, order, order)
            val a: DenseMatrixRegion = if (borrowedOperand == "left") view else matrix(order)
            val b: DenseMatrixRegion = if (borrowedOperand == "right") view else matrix(order)
            val c: DenseMatrixRegion = if (borrowedOperand == "destination") view else matrix(order)
            val before = c.values.copyOf()
            fun copy(region: DenseMatrixRegion) = DenseMatrix.wrap(
                order,
                order,
                DoubleArray(order * order) { region[it % order, it / order] },
            )
            val expected = copy(c)
            ReferenceBlas.gemm(0.875, copy(a), false, copy(b), false, -0.25, expected)

            blas.gemm(0.875, a, false, b, false, -0.25, c, scratch = Scratch())

            for (j in 0 until order) {
                for (i in 0 until order) {
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

        blas.gemv(1.0, a, x, 0, 1, 0.0, y, 0, 1)
        blas.symv(1.0, a, x, 0, 1, 0.0, y, 0, 1)
        blas.ger(1.0, x, 0, 1, x, 0, 1, matrix(ORDER))
        blas.syr(1.0, v.values, v.offset, v.stride, matrix(ORDER))
        blas.syr2(1.0, v.values, v.offset, v.stride, v.values, v.offset, v.stride, matrix(ORDER))
        blas.trsv(a, x.copyOf(), 0, 1, lower = true)
        blas.trmv(a, x.copyOf(), 0, 1, lower = true)
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

        blas.gemv(1.0, panel, DoubleArray(3), 0, 1, 0.0, DoubleArray(5), 0, 1, scratch = workspace)
        blas.gemm(1.0, panel, false, beside, false, 1.0, trailing, scratch = workspace)
        blas.syrk(-1.0, panel, false, 1.0, trailing, scratch = workspace)
        blas.trsv(triangle, DoubleArray(3), 0, 1, lower = true, scratch = workspace)
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
