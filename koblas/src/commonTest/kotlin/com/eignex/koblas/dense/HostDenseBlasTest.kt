package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Which of the two implementations a composed default call reaches, and what it hands over.
 *
 * The binding records and computes nothing, so these ask what was called rather than whether the numbers
 * came back right: a threshold that never fires, a packed layout offered to a library that cannot read it,
 * or an operand handed over while it still shares the destination fails here and nowhere else.
 * `minimumWork = 0` lets a three-by-three fixture stand for a call past the policy's size, which is itself a
 * separate case below.
 */
class HostDenseBlasTest {
    private fun forced(
        recorder: RecordingBlas = RecordingBlas(),
        minimumWork: Long = 0,
    ): Pair<RecordingBlas, HostDenseBlas> =
        recorder to HostDenseBlas(PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels), recorder, minimumWork)

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
    fun `a scaled product reaches the library on the same terms as an unscaled one`() {
        val (recorder, blas) = forced()
        val a = matrix(ORDER)

        blas.gemm(SCALE, a, false, a, false, 0.0, matrix(ORDER))
        val route = blas.routeOf(
            DenseMatrixOperation.Gemm,
            DenseCall(ORDER, ORDER, alpha = SCALE, depth = ORDER),
        )

        assertEquals(listOf(BlasOperation.Gemm), recorder.calls.map { it.operation })
        assertEquals(HOST_SCHEDULING, route.scheduling)
    }

    @Test
    fun `a call below the policy size stays on the portable schedule`() {
        val (recorder, blas) = forced(minimumWork = HostDensePolicy.MINIMUM_WORK)
        val a = matrix(ORDER)
        val c = matrix(ORDER)

        blas.gemm(1.0, a, false, a, false, 0.0, c)
        val route = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = ORDER))

        assertEquals(emptyList(), recorder.calls.map { it.operation }, "a small product reached the library")
        assertEquals(DENSE_SCHEDULING, route.scheduling)
        assertEquals(null, route.host, "a portable route named a host call")
    }

    @Test
    fun `the policy admits a product once its arithmetic is past the fixed size`() {
        val (recorder, blas) = forced(minimumWork = HostDensePolicy.MINIMUM_WORK)
        val order = CROSSING_ORDER
        val a = matrix(order, order, from = 0.0)

        blas.gemm(1.0, a, false, a, false, 0.0, matrix(order, order, from = 0.0))

        assertEquals(listOf(BlasOperation.Gemm), recorder.calls.map { it.operation })
        assertTrue(
            order.toLong() * order * order >= HostDensePolicy.MINIMUM_WORK,
            "the fixture is not actually past the policy's size",
        )
    }

    // A retained panel is grouped for this library's own register tile, which no library has an argument
    // for; the fixture is well past the policy's number, so size is not what settles it.
    @Test
    fun `a product over retained panels never names a library however large it is`() {
        val (_, blas) = forced()
        val call = DenseCall(CROSSING_ORDER, CROSSING_ORDER, depth = CROSSING_ORDER)

        for (operation in PACKED) {
            val route = blas.routeOf(operation, call)

            assertEquals(DENSE_SCHEDULING, route.scheduling, "$operation named a host route")
            assertEquals(null, route.host, "$operation carried a vendor call")
        }
    }

    @Test
    fun `an operation the library does not export stays on the portable schedule`() {
        val without = RecordingBlas(directlyImplemented = BlasOperation.entries.toSet() - BlasOperation.Gemmt)
        val (recorder, blas) = forced(without)
        val a = matrix(ORDER)

        blas.gemmt(1.0, a, false, a, false, 0.0, matrix(ORDER))
        val route = blas.routeOf(DenseMatrixOperation.Gemmt, DenseCall(ORDER, ORDER, depth = ORDER))

        assertEquals(emptyList(), recorder.calls.map { it.operation }, "an unexported entry point was called")
        assertEquals(DENSE_SCHEDULING, route.scheduling)
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
        val zeroAlpha = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, alpha = 0.0, depth = ORDER))
        val zeroDepth = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = 0))

        assertEquals(emptyList(), recorder.calls.map { it.operation }, "a no-work call reached the library")
        assertEquals(RouteKind.NoWork, zeroAlpha.kind)
        assertEquals(RouteKind.NoWork, zeroDepth.kind)
        assertEquals(null, zeroAlpha.host)
    }

    // A route with no shared dimension is unstated rather than small, so a host branch answering from the
    // other two extents would describe a product nobody described.
    @Test
    fun `a level three route with no shared dimension is refused rather than answered`() {
        val (_, blas) = forced()

        for (operation in NEEDS_DEPTH) {
            assertFailsWith<IllegalArgumentException>("$operation answered without a shared dimension") {
                blas.routeOf(operation, DenseCall(CROSSING_ORDER, CROSSING_ORDER))
            }
        }
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
    fun `the staged copy is borrowed from the workspace of the caller`() {
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
    fun `a route names the staging a shared buffer forces and an ordinary call names none`() {
        val (_, blas) = forced()
        val call = DenseCall(ORDER, ORDER, depth = ORDER)

        val plain = blas.routeOf(DenseMatrixOperation.Gemm, call)
        val staged = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = ORDER, aliased = true))

        assertEquals(listOf("openblas/${BlasOperation.Gemm.entryPoint}"), plain.components)
        assertEquals(listOf(HOST_STAGING, "openblas/${BlasOperation.Gemm.entryPoint}"), staged.components)
    }

    @Test
    fun `a host route carries the library that would run it`() {
        val (recorder, blas) = forced()

        val route = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = ORDER))

        assertEquals(HOST_SCHEDULING, route.scheduling)
        assertEquals(RouteKind.Direct, route.kind)
        assertEquals(recorder.vendor, route.host?.vendor)
        assertEquals(BlasOperation.Gemm.entryPoint, route.host?.entryPoint)
        assertTrue(recorder.libraryPath in (route.reason ?: ""), "the route does not name the resolved binary")
        assertTrue(recorder.version in (route.reason ?: ""), "the route does not name the library's version")
    }

    private companion object {
        /** Small enough that only a forced policy sends it over, which is what the routing cases want. */
        const val ORDER = 3

        /** Past [HostDensePolicy.MINIMUM_WORK] as a cubic product, and still a fixture a test can hold. */
        const val CROSSING_ORDER = 64

        const val SCALE = 0.75

        val PACKED = listOf(
            DenseMatrixOperation.GemmPacked,
            DenseMatrixOperation.GemmPackedLeft,
            DenseMatrixOperation.GemmPackedRight,
        )

        val NEEDS_DEPTH = listOf(
            DenseMatrixOperation.Gemm,
            DenseMatrixOperation.Gemmt,
            DenseMatrixOperation.Symm,
            DenseMatrixOperation.Syrk,
            DenseMatrixOperation.Syr2k,
            DenseMatrixOperation.Trmm,
            DenseMatrixOperation.Trsm,
        )
    }
}
