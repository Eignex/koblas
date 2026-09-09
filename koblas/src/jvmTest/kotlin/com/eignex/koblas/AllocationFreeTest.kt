package com.eignex.koblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.sparseConformanceSystem
import com.eignex.koblas.testutil.allocation.allocatedBytes
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.random.Random
import kotlin.test.*

class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        /** A pooled form must allocate at most this fraction of what the allocating form does. */
        const val POOLED_RATIO = 50.0
    }

    /** Pins the portable backend for the suite, since an FFM call allocates a MemorySegment wrapper per array. */
    @BeforeTest
    fun usePortableKernels() {
        installBackends(
            koblas.with(
                kernels = ScalarKernels,
                blas = F64ReferenceBlas,
                sparseBlas = F64ReferenceSparseLinearAlgebra,
                sparseDecompositions = F64ReferenceSparseLinearAlgebra,
                sparseKernels = F64ReferenceSparseLinearAlgebra,
            ),
        )
    }

    @AfterTest
    fun restoreSelection() {
        installBackends(null)
    }

    private fun assertPooled(pooled: Double, allocating: Double, what: String) {
        val budget = maxOf(FLOOR_BYTES, allocating / POOLED_RATIO)
        assertTrue(
            pooled < budget,
            "$what allocated $pooled B per iteration against a $budget B budget (allocating form: $allocating B)",
        )
    }

    @Test
    fun `reserved workspace allocates nothing on first borrows`() {
        val count = 16
        val warm = Workspace().apply {
            reserve(8, count)
            reserveI32(8, count)
        }
        val warmDoubles = arrayOfNulls<DoubleArray>(count)
        val warmIndices = arrayOfNulls<IntArray>(count)
        repeat(200) {
            for (i in 0 until count) warmDoubles[i] = warm.take(8)
            for (i in 0 until count) warmIndices[i] = warm.takeI32(8)
            for (i in 0 until count) warm.release(requireNotNull(warmDoubles[i]))
            for (i in 0 until count) warm.release(requireNotNull(warmIndices[i]))
        }

        val smallCount = 2
        val smallWorkspace = Workspace().apply {
            reserve(8, smallCount)
            reserveI32(8, smallCount)
        }
        val smallDoubles = arrayOfNulls<DoubleArray>(smallCount)
        val smallIndices = arrayOfNulls<IntArray>(smallCount)
        val measurementFloor = allocatedBytes { Unit }
        val smallBytes = allocatedBytes {
            for (i in 0 until smallCount) smallDoubles[i] = smallWorkspace.take(8)
            for (i in 0 until smallCount) smallIndices[i] = smallWorkspace.takeI32(8)
            for (i in 0 until smallCount) smallWorkspace.release(requireNotNull(smallDoubles[i]))
            for (i in 0 until smallCount) smallWorkspace.release(requireNotNull(smallIndices[i]))
        }

        val workspace = Workspace().apply {
            reserve(8, count)
            reserveI32(8, count)
        }
        val doubles = arrayOfNulls<DoubleArray>(count)
        val indices = arrayOfNulls<IntArray>(count)
        val bytes = allocatedBytes {
            for (i in 0 until count) doubles[i] = workspace.take(8)
            for (i in 0 until count) indices[i] = workspace.takeI32(8)
            for (i in 0 until count) workspace.release(requireNotNull(doubles[i]))
            for (i in 0 until count) workspace.release(requireNotNull(indices[i]))
        }

        assertTrue(
            smallBytes <= measurementFloor,
            "small reserved first borrows allocated $smallBytes bytes against a $measurementFloor byte floor",
        )
        assertTrue(
            bytes <= measurementFloor,
            "reserved first borrows allocated $bytes bytes against a $measurementFloor byte measurement floor",
        )
    }

    @Test
    fun `norm1 allocates nothing`() {
        val n = 64
        val a = wellConditioned(n, Random(20260742))
        val bytes = bytesPerIteration(2000) { a.norm1() }
        assertTrue(bytes <= FLOOR_BYTES, "norm1 allocated $bytes B")
    }

    @Test
    fun `a sparse simplex-shaped iteration allocates nothing`() {
        val m = 64
        val rng = Random(20260739)
        val columns = List(m) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-1.0, 1.0) + m))
            for (i in 0 until m) if (i != j && rng.nextDouble() < 0.05) entries.add(i to rng.nextDouble(-1.0, 1.0))
            entries
        }
        val basis = SparseMatrix.ofColumns(m, m, columns).lu()
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(m)
        val y = DoubleArray(m)
        val ws = Workspace()

        val allocating = bytesPerIteration(500) {
            basis.solve(b)
            basis.solve(b, transpose = true)
        }
        val into = bytesPerIteration(500) {
            basis.solveInto(b, x, workspace = ws)
            basis.solveInto(b, y, transpose = true, workspace = ws)
        }
        assertTrue(allocating > m * Double.SIZE_BYTES * 2.0, "expected allocation, saw $allocating B")
        assertPooled(into, allocating, "sparse solve both directions")
    }

    @Test
    fun `strict sparse allocation checks are allocation neutral`() {
        val n = 64
        val factor = sparseConformanceSystem(n, Random(20260828)).lu()
        val b = DoubleArray(n) { it * 0.01 - 0.5 }
        val out = DoubleArray(n)
        val workspace = Workspace().apply { reserve(n, count = 2) }

        val bytes = bytesPerIteration(500) {
            factor.solveInto(
                b,
                out,
                workspace = workspace,
                allocationPolicy = AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE,
            )
        }

        assertTrue(bytes <= FLOOR_BYTES, "strict sparse solve allocated $bytes B per call")
    }

    /**
     * The symmetric rank-one and rank-two updates take a [VectorLike], and densifying a dense operand to
     * read it copies the whole vector on every call. They are the innermost step of a covariance or a
     * quasi-Newton update, so a copy per call is a copy per iteration of the caller's loop.
     */
    @Test
    fun `a symmetric rank update loop allocates nothing per iteration`() {
        val n = 96
        val rng = Random(20260824)
        val a = DenseMatrix.zero(n, n)
        val x = DenseVector.wrap(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
        val y = DenseVector.wrap(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
        val syr = bytesPerIteration(500) { koblas.syr(1e-12, x, a, lower = true) }
        val syr2 = bytesPerIteration(500) { koblas.syr2(1e-12, x, y, a, lower = true) }
        assertTrue(syr < FLOOR_BYTES, "syr allocated $syr B per call for a dense operand it can read in place")
        assertTrue(syr2 < FLOOR_BYTES, "syr2 allocated $syr2 B per call for dense operands it can read in place")
    }

    @Test
    fun `a transposed dense gemv workspace is allocation neutral`() {
        val rows = 64
        val columns = 72
        val a = DenseMatrix(rows, columns)
        val x = DoubleArray(rows) { it * 0.01 }
        val y = DoubleArray(columns)
        val workspace = Workspace().apply { reserve(4, count = 1) }

        val bytes = bytesPerIteration(1_000) {
            koblas.gemv(1e-8, a, x, 1.0, y, transpose = true, workspace = workspace)
            y
        }

        assertTrue(bytes <= FLOOR_BYTES, "transposed gemv allocated $bytes B per call")
    }

    @Test
    fun `transposed dense gemm workspace is allocation neutral`() {
        val m = 32
        val k = 48
        val n = 24
        val a = DenseMatrix(k, m)
        val b = DenseMatrix(n, k)
        val c = DenseMatrix(m, n)
        val workspace = Workspace().apply {
            reserve(k * n, count = 1)
            reserve(4, count = 1)
        }

        val bytes = bytesPerIteration(300) {
            koblas.gemm(1e-8, a, true, b, true, 1.0, c, workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "transposed gemm allocated $bytes B per call")
    }

    @Test
    fun `symmetric matrix workspace is allocation neutral on both sides`() {
        val n = 96
        val width = 12
        val a = DenseMatrix.diagonal(n)
        for (right in booleanArrayOf(false, true)) {
            val b = if (right) DenseMatrix(width, n) else DenseMatrix(n, width)
            val c = DenseMatrix(b.rows, b.cols)
            val workspace = Workspace()

            val bytes = bytesPerIteration(300) {
                koblas.symm(1e-8, a, b, 1.0, c, right = right, workspace = workspace)
                c
            }

            assertTrue(bytes <= FLOOR_BYTES, "symm right=$right allocated $bytes B per call")
        }
    }

    @Test
    fun `right dense triangular solve workspace is allocation neutral`() {
        val n = 80
        val triangle = DenseMatrix.diagonal(n)
        val b = DenseMatrix(20, n)
        val workspace = Workspace().apply { reserve(n, count = 1) }

        val bytes = bytesPerIteration(500) {
            triangle.trsm(b, lower = true, right = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "right dense trsm allocated $bytes B per call")
    }

    @Test
    fun `sparse dense product workspace is allocation neutral`() {
        val n = 64
        val rows = 16
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = DenseMatrix(n, rows)
        val c = DenseMatrix(rows, n)
        val workspace = Workspace().apply { reserve(b.data.size, count = 1) }

        val bytes = bytesPerIteration(300) {
            koblas.sparseBlas.gemm(1e-8, sparse, false, b, true, 1.0, c, right = true, workspace = workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "right transposed sparse gemm allocated $bytes B per call")
    }

    @Test
    fun `left sparse triangular solve workspace is allocation neutral`() {
        val n = 64
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = DenseMatrix(n, 12)
        val workspace = Workspace().apply {
            reserve(n, count = 1)
            reserve(REFERENCE_SPARSE_RHS_WIDTH, count = 1)
        }

        val bytes = bytesPerIteration(500) {
            koblas.sparseBlas.trsm(sparse, b, lower = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "left sparse trsm allocated $bytes B per call")
    }

    @Test
    fun `inspecting a route reads one role rather than the whole snapshot`() {
        // route() took the whole twelve-role status to read one entry, so every operation under a strict
        // dispatch policy built twelve BackendStatus objects, up to twelve BackendMetadata, a list and two
        // sets. Nothing in this suite covered it: the other cases all run the default AUTO context, where
        // the policy path never executes.
        val query = F64RouteQuery.DenseGemv(64, 64)

        val single = bytesPerIteration(2000) { koblas.route(query) }
        val whole = bytesPerIteration(2000) { koblas.status }

        // Not allocation-free, and cannot be: route returns a BackendRoute over a BackendStatus, so one of
        // each is the floor. What it must not do is build the other eleven roles to get there.
        assertTrue(whole > FLOOR_BYTES, "expected the full snapshot to allocate, saw $whole B")
        assertTrue(
            single * 4 < whole,
            "route allocated $single B against the full snapshot's $whole B, so it is still building all of it",
        )
    }
}
