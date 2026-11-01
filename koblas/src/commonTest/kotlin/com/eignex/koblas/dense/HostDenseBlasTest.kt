package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseMatrixRegion
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.dense.PortableDenseBlas
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Checks supplied eligibility, symbol fallback, aliases and argument translation with a recording binding.
 * Allowing small fixtures isolates composition from platform cutoffs; EngineProfileTest cover those cutoffs.
 */
class HostDenseBlasTest {
    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertClose(expected, actual, context = context)

    @Test
    fun `platform eligibility keeps a rejected host call portable`() {
        val recorder = RecordingBlas()
        val portable = PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels)
        val blas = PortableDenseBlas(portable, recorder, eligible = false)
        val a = matrix(ORDER)
        val actual = matrix(ORDER)
        val expected = matrix(ORDER)
        ReferenceBlas.gemm(0.875, a, false, a, false, -0.25, expected)

        blas.gemm(0.875, a, false, a, false, -0.25, actual)

        assertTrue(recorder.calls.isEmpty())
        assertAgreesWithReference(expected.values, actual.values, "host eligibility fallback")
    }

    @Test
    fun `borrowed eligibility rejects any adapted operand before staging`() {
        for (borrowedOperand in listOf("left", "right", "destination")) {
            val recorder = RecordingBlas()
            val portable = PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels)
            val blas = PortableDenseBlas(portable, recorder, borrowed = false)
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

            blas.gemm(0.875, a, false, b, false, -0.25, c, Workspace())

            for (j in 0 until ORDER) {
                for (i in 0 until ORDER) {
                    before[c.offset + i + j * c.leadingDimension] = expected[i, j]
                }
            }
            assertTrue(recorder.calls.isEmpty(), borrowedOperand)
            assertAgreesWithReference(before, c.values, "$borrowedOperand region eligibility fallback")
        }
    }

    private fun forced(recorder: RecordingBlas = RecordingBlas()): Pair<RecordingBlas, PortableDenseBlas> =
        recorder to PortableDenseBlas(
            PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels),
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
            BlasOperation.Gemv, BlasOperation.Symv, BlasOperation.Ger, BlasOperation.Syr, BlasOperation.Syr2,
            BlasOperation.Trsv, BlasOperation.Trmv, BlasOperation.Gemm, BlasOperation.Gemmt, BlasOperation.Syrk,
            BlasOperation.Syr2k, BlasOperation.Symm, BlasOperation.Trsm, BlasOperation.Trmm,
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

        assertEquals(listOf(BlasOperation.Gemm), recorder.calls.map { it.operation })
    }

    @Test
    fun `an operation the library does not export stays on the portable schedule`() {
        val without = RecordingBlas(directlyImplemented = BlasOperation.entries.toSet() - BlasOperation.Gemmt)
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
    fun `an input sharing the destination is copied before the library sees it`() {
        val (recorder, blas) = forced()
        val shared = DoubleArray(ORDER * ORDER) { 1.0 + it }
        val a = DenseMatrix.wrap(ORDER, ORDER, shared)
        val c = DenseMatrix.wrap(ORDER, ORDER, shared)

        blas.gemm(1.0, a, false, a, false, 0.0, c)

        val call = recorder.single()
        assertNotSame(shared, call.matrices[0].values, "the left operand still shared the destination's buffer")
        assertNotSame(shared, call.matrices[1].values, "the right operand still shared the destination's buffer")
        assertSame(shared, call.matrices[2].values, "the destination was not the caller's own storage")
        assertContentEquals(
            DoubleArray(ORDER * ORDER) { 1.0 + it },
            call.matrices[0].values,
            "the staged copy does not hold what the operand held",
        )
    }

    @Test
    fun `shared operands reach the library through one snapshot`() {
        for (operation in listOf(BlasOperation.Gemm, BlasOperation.Gemmt, BlasOperation.Symm, BlasOperation.Syr2k)) {
            for (sameMatrix in booleanArrayOf(false, true)) {
                for (useWorkspace in booleanArrayOf(false, true)) {
                    val (recorder, blas) = forced()
                    val c = matrix(ORDER)
                    val b = if (sameMatrix) c else DenseMatrix.wrap(ORDER, ORDER, c.values)
                    val workspace = if (useWorkspace) Workspace() else null

                    when (operation) {
                        BlasOperation.Gemm -> blas.gemm(0.875, c, false, b, true, -0.25, c, workspace)
                        BlasOperation.Gemmt -> blas.gemmt(0.875, c, false, b, true, -0.25, c, workspace = workspace)
                        BlasOperation.Symm -> blas.symm(0.875, c, b, -0.25, c, workspace = workspace)
                        BlasOperation.Syr2k -> blas.syr2k(0.875, c, b, true, -0.25, c, workspace = workspace)
                        else -> error("unexpected product $operation")
                    }

                    val call = recorder.single()
                    assertEquals(operation, call.operation)
                    assertSame(call.matrices[0].values, call.matrices[1].values, "$operation duplicated its snapshot")
                    assertNotSame(c.values, call.matrices[0].values, "$operation did not stage the shared input")
                    assertContentEquals(c.values, call.matrices[0].values)
                    workspace?.let { assertEquals(1, it.available(ORDER * ORDER)) }
                }
            }
        }
    }

    @Test
    fun `host product aliases agree with the reference and share one snapshot`() {
        val portable = PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels)
        val blas = PortableDenseBlas(portable, ReferenceProductBlas())

        assertDenseProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `host rectangular product aliases retain their operand shapes`() {
        val portable = PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels)
        val blas = PortableDenseBlas(portable, ReferenceProductBlas())

        assertRectangularProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `the staged copy is borrowed from the caller s workspace`() {
        val (_, blas) = forced()
        val workspace = Workspace()
        val shared = DoubleArray(ORDER * ORDER) { 1.0 + it }
        val c = DenseMatrix.wrap(ORDER, ORDER, shared)

        blas.gemm(1.0, DenseMatrix.wrap(ORDER, ORDER, shared), false, matrix(ORDER), false, 0.0, c, workspace)
        val lent = workspace.available(ORDER * ORDER)
        blas.gemm(1.0, DenseMatrix.wrap(ORDER, ORDER, shared), false, matrix(ORDER), false, 0.0, c, workspace)

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

        blas.gemm(1.0, a, false, b, false, 0.0, c, Workspace())

        val call = recorder.single()
        assertEquals(BlasOperation.Gemm, call.operation)
        for (i in expectedA.indices) assertContentEquals(expectedA[i], call.matrices[0].toArray()[i])
        for (i in expectedB.indices) assertContentEquals(expectedB[i], call.matrices[1].toArray()[i])
        assertNotSame(parent.values, call.matrices[0].values)
    }

    @Test
    fun `borrowed host calls reach a region binding in place`() {
        val recorder = RecordingBlas()
        val blas = PortableDenseBlas(
            PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels),
            recorder,
        )
        val parent = matrix(9)
        val panel = parent.view(4, 0, 5, 3)
        val trailing = parent.view(4, 4, 5, 5)
        val triangle = parent.view(0, 0, 3, 3)
        val beside = parent.view(0, 4, 3, 5)
        val workspace = Workspace()

        blas.gemv(1.0, panel, DoubleArray(3), 0.0, DoubleArray(5), workspace = workspace)
        blas.gemm(1.0, panel, false, beside, false, 1.0, trailing, workspace)
        blas.syrk(-1.0, panel, false, 1.0, trailing, workspace = workspace)
        blas.trsv(triangle, DoubleArray(3), lower = true, workspace = workspace)
        blas.trsm(triangle, beside, lower = true, workspace = workspace)

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
        val blas = PortableDenseBlas(
            PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels),
            recorder,
        )
        val parent = matrix(7)
        val a = parent.view(0, 0, 4, 3)
        val c = parent.view(1, 1, 4, 4)
        val entries = a.toArray()

        blas.syrk(-1.0, a, false, 1.0, c, workspace = Workspace())

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
private class ReferenceProductBlas : Blas by RecordingBlas() {
    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ) = ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c)

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = ReferenceBlas.gemmt(alpha, a, transposeA, b, transposeB, beta, c, structure == MatrixStructure.SymmetricLower)

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ) = ReferenceBlas.symm(alpha, a, b, beta, c, structure == MatrixStructure.SymmetricLower, rightSide)

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) = ReferenceBlas.syr2k(alpha, a, b, transposeA, beta, c, structure == MatrixStructure.SymmetricLower)
}
