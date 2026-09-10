package com.eignex.koblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH
import com.eignex.koblas.testutil.allocation.allocatedBytes
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.random.Random
import kotlin.test.*

class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        val engine = BuiltinKernels.scalar
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
        val syr = bytesPerIteration(500) { engine.syr(1e-12, x, a, lower = true) }
        val syr2 = bytesPerIteration(500) { engine.syr2(1e-12, x, y, a, lower = true) }
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
            engine.gemv(1e-8, a, x, 1.0, y, transpose = true, workspace = workspace)
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
            engine.gemm(1e-8, a, true, b, true, 1.0, c, workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "transposed gemm allocated $bytes B per call")
    }

    @Test
    fun `triangular result gemm workspace is allocation neutral`() {
        val n = 64
        val k = 47
        val a = DenseMatrix(n, k)
        val b = DenseMatrix(k, n)
        val c = DenseMatrix(n)
        val workspace = Workspace()
        engine.gemmt(1e-8, a, false, b, false, 1.0, c, workspace = workspace)

        val bytes = bytesPerIteration(300) {
            engine.gemmt(1e-8, a, false, b, false, 1.0, c, workspace = workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "gemmt allocated $bytes B per call")
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
                engine.symm(1e-8, a, b, 1.0, c, right = right, workspace = workspace)
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
            engine.trsm(triangle, b, lower = true, right = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "right dense trsm allocated $bytes B per call")
    }

    @Test
    fun `packed solve kernels allocate nothing`() {
        val rows = koblas.packedPanels.tileRows
        val order = koblas.packedPanels.tileColumns
        val depth = 32
        val left = DoubleArray(koblas.packedPanels.leftSize(rows, depth)) { 0.01 * (it + 1) }
        val right = DoubleArray(koblas.packedPanels.rightSize(depth, order)) { 0.005 * (it + 1) }
        val triangle = DoubleArray(koblas.packedPanels.rightSize(order, order))
        for (i in 0 until order) {
            for (j in 0..i) triangle[i * order + j] = if (i == j) 2.0 else 0.1
        }
        val x = DoubleArray(koblas.packedPanels.leftSize(rows, order)) { 1.0 }

        val solveBytes = bytesPerIteration(1000) {
            koblas.packedPanels.trsm(triangle, x, rows, order, lower = true)
            x
        }
        val fusedBytes = bytesPerIteration(1000, warmup = 20_000) {
            koblas.packedPanels.gemmTrsm(left, right, triangle, x, rows, order, depth, lower = true)
            x
        }

        assertTrue(solveBytes <= FLOOR_BYTES, "packed trsm allocated $solveBytes B per call")
        assertTrue(fusedBytes <= FLOOR_BYTES, "packed gemm trsm allocated $fusedBytes B per call")
    }

    @Test
    fun `ordinary packed trsm allocates nothing with reserved workspace`() {
        val order = DenseTuning.trsmPackedMinOrder
        val rows = DenseTuning.trsmPackedMinRows
        val triangle = DenseMatrix.zero(order)
        for (column in 0 until order) {
            for (row in column until order) triangle[row, column] = if (row == column) 1.0 else 1e-12
        }
        val rightHandSide = DenseMatrix.zero(rows, order)
        val packedTriangleSize = packedRightSize(order, order, PortablePackedKernels.gemmTileCols)
        val packedRightHandSideSize = packedLeftSize(rows, order, PortablePackedKernels.gemmTileRows)
        val workspace = Workspace().apply {
            reserve(maxOf(packedTriangleSize, packedRightHandSideSize), count = 2)
        }

        val bytes = bytesPerIteration(500) {
            engine.trsm(triangle, rightHandSide, lower = true, right = true, workspace = workspace)
            rightHandSide
        }

        assertTrue(bytes <= FLOOR_BYTES, "ordinary packed trsm allocated $bytes B per call")
    }

    @Test
    fun `ordinary packed trmm allocates nothing with reserved workspace`() {
        val order = DenseTuning.trmmPackedMinOrder
        val panel = DenseTuning.trmmPackedMinRows
        val triangle = DenseMatrix.diagonal(order)
        val rightHandSide = DenseMatrix(panel, order, DoubleArray(panel * order) { 1e-4 * (it + 1) })
        val largestPackedPanel = maxOf(
            rightHandSide.data.size,
            DenseTuning.packedBlockRows * minOf(DenseTuning.packedBlockDepth, order),
            minOf(DenseTuning.packedBlockDepth, order) * DenseTuning.packedBlockColumns,
            PortablePackedKernels.gemmTileRows * PortablePackedKernels.gemmTileCols,
        )
        val workspace = Workspace().apply { reserve(largestPackedPanel, count = 4) }

        val bytes = bytesPerIteration(500) {
            engine.trmm(triangle, rightHandSide, lower = true, right = true, workspace = workspace)
            rightHandSide
        }

        assertTrue(bytes <= FLOOR_BYTES, "ordinary packed trmm allocated $bytes B per call")
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
            engine.gemm(1e-8, sparse, false, b, true, 1.0, c, right = true, workspace = workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "right transposed sparse gemm allocated $bytes B per call")
    }

    @Test
    fun `sparse vector and matrix slice kernels allocate nothing`() {
        val n = 128
        val sparse = SparseMatrix.ofColumns(
            n,
            n,
            List(n) { column -> listOf(column to 1.0, (column + 1) % n to 1e-8).sortedBy { it.first } },
        )
        val vector = SparseVector.of(n, IntArray(n / 2) { it * 2 }, DoubleArray(n / 2) { it + 1.0 })
        val x = DoubleArray(n) { it * 0.01 }
        val y = DoubleArray(n)

        val levelOneBytes = bytesPerIteration(1_000) { engine.sparseKernels.dot(vector, x) }
        val gemvBytes = bytesPerIteration(1_000) {
            engine.gemv(1e-12, sparse, x, 1.0, y, transpose = true)
            y
        }
        val symvBytes = bytesPerIteration(1_000) {
            engine.symv(1e-12, sparse, x, 1.0, y, lower = true)
            y
        }

        assertTrue(levelOneBytes <= FLOOR_BYTES, "sparse level one allocated $levelOneBytes B per call")
        assertTrue(gemvBytes <= FLOOR_BYTES, "sparse gemv allocated $gemvBytes B per call")
        assertTrue(symvBytes <= FLOOR_BYTES, "sparse symv allocated $symvBytes B per call")
    }

    @Test
    fun `new sparse dense destinations reuse workspace`() {
        val n = 64
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val identity = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val rhs = DenseMatrix(n, 8)
        val product = DenseMatrix(n)
        val symmetricProduct = DenseMatrix(n, 8)
        val workspace = Workspace().apply {
            reserve(n, count = 1)
            reserveI32(n, count = 2)
        }
        engine.syrk(1e-12, sparse, false, 1.0, product, workspace = workspace)

        val syrkBytes = bytesPerIteration(300) {
            engine.syrk(1e-12, sparse, false, 1.0, product, workspace = workspace)
            product
        }
        val symmBytes = bytesPerIteration(300) {
            engine.symm(1e-12, sparse, rhs, 1.0, symmetricProduct, workspace = workspace)
            symmetricProduct
        }
        val gemmBytes = bytesPerIteration(300) {
            engine.gemm(1e-12, sparse, false, identity, false, 1.0, product, workspace)
            product
        }

        assertTrue(syrkBytes <= FLOOR_BYTES, "sparse syrk allocated $syrkBytes B per call")
        assertTrue(symmBytes <= FLOOR_BYTES, "sparse symm allocated $symmBytes B per call")
        assertTrue(gemmBytes <= FLOOR_BYTES, "direct sparse dense gemm allocated $gemmBytes B per call")
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
            engine.trsm(sparse, b, lower = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "left sparse trsm allocated $bytes B per call")
    }
}
