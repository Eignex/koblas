package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The seams a caller may run inside a hot loop without the collector noticing: Level 1 and the sparse
 * primitives unconditionally, and the matrix algorithms given a workspace to take their scratch from. An
 * operation that discovers a new structure allocates its result by definition and is not measured here.
 */
class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        /** An array object's own bytes, beside the elements it holds. */
        const val ARRAY_HEADER_BYTES = 16

        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
    }

    @Test
    fun `dense level one kernels allocate nothing`() {
        val n = 512
        val a = DoubleArray(n) { it * 0.01 }
        val b = DoubleArray(n) { 1.0 / (it + 1) }
        val kernels = engine.vectorKernels

        val dot = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.dot(a, 0, b, 0, n) }
        val axpy = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.axpy(b, 0, 0.5, a, 0, n) }
        val nrm2 = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.nrm2(a, 0, n) }
        val strided = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.dot(a, 0, b, 0, n / 2, 2, 2) }

        assertTrue(dot <= FLOOR_BYTES, "dot allocated $dot B per call")
        assertTrue(axpy <= FLOOR_BYTES, "axpy allocated $axpy B per call")
        assertTrue(nrm2 <= FLOOR_BYTES, "nrm2 allocated $nrm2 B per call")
        assertTrue(strided <= FLOOR_BYTES, "a strided dot allocated $strided B per call")
    }

    /**
     * The dense Level 2 callers a user writes, on the portable panels. Kover's instrumentation keeps HotSpot
     * from scalar-replacing a Vector API carrier, so the vector panels are measured by the
     * `simdDenseAllocationCheck` task instead; what this covers is the scheduling around them.
     */
    @Test
    fun `dense level two calls allocate nothing`() {
        val n = 48
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until n) triangle.values[i + i * n] = 4.0
        val x = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(n) { 0.5 }
        val view = DenseVector.wrap(x)
        val portable = BuiltinEngines.scalar

        val gemv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.gemv(1e-12, a, x, 1.0, y)
            y
        }
        val transposed = bytesPerIteration(400, FLOOR_BYTES) {
            portable.gemv(1e-12, a, x, 1.0, y, transpose = true)
            y
        }
        val symv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.symv(1e-12, a, x, 1.0, y)
            y
        }
        val ger = bytesPerIteration(400, FLOOR_BYTES) {
            portable.ger(1e-12, x, y, a)
            a
        }
        val syr = bytesPerIteration(400, FLOOR_BYTES) {
            portable.syr(1e-12, view, a)
            a
        }
        val trmv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.trmv(triangle, y, lower = true)
            y
        }
        val trsv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.trsv(triangle, y, lower = true)
            y
        }

        assertTrue(gemv <= FLOOR_BYTES, "gemv allocated $gemv B per call")
        assertTrue(transposed <= FLOOR_BYTES, "transposed gemv allocated $transposed B per call")
        assertTrue(symv <= FLOOR_BYTES, "symv allocated $symv B per call")
        assertTrue(ger <= FLOOR_BYTES, "ger allocated $ger B per call")
        assertTrue(syr <= FLOOR_BYTES, "syr allocated $syr B per call")
        assertTrue(trmv <= FLOOR_BYTES, "trmv allocated $trmv B per call")
        assertTrue(trsv <= FLOOR_BYTES, "trsv allocated $trsv B per call")
    }

    /** The convenience callers, which pass a contiguous vector through as the array it already is. */
    @Test
    fun `contiguous convenience calls allocate nothing`() {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val contiguous = DenseVector.wrap(DoubleArray(n) { 1.0 + (it % 7) * 0.25 })
        val destination = DoubleArray(n)

        val gemv = bytesPerIteration(1_000, FLOOR_BYTES) {
            a.gemvInto(1e-12, contiguous, 1.0, destination)
            destination
        }
        val symv = bytesPerIteration(1_000, FLOOR_BYTES) {
            a.symvInto(1e-12, contiguous, 1.0, destination)
            destination
        }

        assertTrue(gemv <= FLOOR_BYTES, "gemvInto allocated $gemv B per call")
        assertTrue(symv <= FLOOR_BYTES, "symvInto allocated $symv B per call")
    }

    /**
     * A strided operand is gathered into an array, which is measured rather than wished away. The bound is
     * an upper one, because addressing a step without gathering would be an improvement.
     */
    @Test
    fun `a strided convenience operand is gathered once per call`() {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val buffer = DoubleArray(2 * n) { 1.0 + (it % 7) * 0.25 }
        val strided = StridedVector(buffer, 0, n, 2)
        val destination = DoubleArray(n)
        val oneGather = n * Double.SIZE_BYTES + ARRAY_HEADER_BYTES

        val gathered = bytesPerIteration(1_000, oneGather.toDouble()) {
            a.gemvInto(1e-12, strided, 1.0, destination)
            destination
        }

        assertTrue(gathered <= oneGather + FLOOR_BYTES, "a strided gemvInto allocated $gathered B per call")
    }

    /**
     * A dense Level 3 call lent a workspace. Without one these allocate by design: a solve gathers each
     * right-hand side and a multiply copies it again as its source.
     */
    @Test
    fun `dense level three calls reuse a workspace`() {
        val order = 32
        val sides = 8
        val portable = BuiltinEngines.scalar
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until order) triangle.values[i + i * order] = 4.0
        val b = DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { 1.0 })
        val workspace = Workspace()
        portable.trsm(triangle, b, lower = true, workspace = workspace)
        portable.trmm(triangle, b, lower = true, workspace = workspace)

        val solve = bytesPerIteration(100, FLOOR_BYTES) {
            portable.trsm(triangle, b, lower = true, workspace = workspace)
            b
        }
        val multiply = bytesPerIteration(100, FLOOR_BYTES) {
            portable.trmm(triangle, b, lower = true, workspace = workspace)
            b
        }

        assertTrue(solve <= FLOOR_BYTES, "trsm with a workspace allocated $solve B per call")
        assertTrue(multiply <= FLOOR_BYTES, "trmm with a workspace allocated $multiply B per call")
    }

    /**
     * The matrix product on both of its routes, at a depth that fits one block and one that does not. The
     * blocked route's two packed panels are loans; a retained pair borrows nothing, having no copy left.
     */
    @Test
    fun `matrix products reuse a workspace at both depths`() {
        val portable = BuiltinEngines.scalar
        val workspace = Workspace()
        val wide = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48) { 1.0 + (it % 13) * 0.125 })
        val deepLeft = DenseMatrix.wrap(12, 400, DoubleArray(12 * 400) { 1.0 + (it % 7) * 0.25 })
        val deepRight = DenseMatrix.wrap(400, 12, DoubleArray(400 * 12) { 0.5 + (it % 5) * 0.125 })
        val deepTarget = DenseMatrix.wrap(12, 12, DoubleArray(144))
        val small = DenseMatrix.wrap(5, 5, DoubleArray(25) { 1.0 + it })
        val smallTarget = DenseMatrix.wrap(5, 5, DoubleArray(25))
        val left = portable.packLeft(wide, transpose = false)
        val right = portable.packRight(wide, transpose = false)
        val packedTarget = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48))
        portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
        portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)

        val shallow = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
            packedTarget
        }
        val deep = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)
            deepTarget
        }
        val retained = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, left, right, 1.0, packedTarget)
            packedTarget
        }
        portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
        val panelRoute = bytesPerIteration(200, FLOOR_BYTES) {
            portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
            smallTarget
        }

        assertTrue(shallow <= FLOOR_BYTES, "a blocked product allocated $shallow B per call")
        assertTrue(deep <= FLOOR_BYTES, "a product over several depth blocks allocated $deep B per call")
        assertTrue(retained <= FLOOR_BYTES, "a retained product allocated $retained B per call")
        assertTrue(panelRoute <= FLOOR_BYTES, "an unpacked product allocated $panelRoute B per call")
    }

    /**
     * The one buffer an unpacked product takes when it is lent no workspace: the destination column it
     * accumulates before spending the multipliers, which is what keeps alpha on a sum of products.
     */
    @Test
    fun `an unpacked product without a workspace takes one destination column`() {
        val order = 64
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(order, 2, DoubleArray(order * 2) { 1.0 + (it % 13) * 0.125 })
        val b = DenseMatrix.wrap(2, 2, DoubleArray(4) { 0.5 })
        val c = DenseMatrix.wrap(order, 2, DoubleArray(order * 2))
        val oneColumn = order * Double.SIZE_BYTES + ARRAY_HEADER_BYTES

        val bytes = bytesPerIteration(500, oneColumn.toDouble()) {
            portable.gemm(1e-12, a, false, b, false, 1.0, c)
            c
        }

        assertTrue(bytes <= oneColumn + FLOOR_BYTES, "an unpacked product allocated $bytes B per call")
    }

    @Test
    fun `sparse vector kernels allocate nothing`() {
        val n = 128
        val vector = SparseVector.of(n, IntArray(n / 2) { it * 2 }, DoubleArray(n / 2) { it + 1.0 })
        val x = DoubleArray(n) { it * 0.01 }

        val levelOneBytes = bytesPerIteration(1_000, FLOOR_BYTES) { engine.sparseKernels.dot(vector, x) }

        assertTrue(levelOneBytes <= FLOOR_BYTES, "sparse level one allocated $levelOneBytes B per call")
    }

    @Test
    fun `sparse matrix-vector calls allocate nothing`() {
        val n = 128
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val x = DoubleArray(n) { it * 0.01 }
        val y = DoubleArray(n)

        val gemv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.gemv(1e-12, sparse, x, 1.0, y, transpose = true)
            y
        }
        val symv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.symv(1e-12, sparse, x, 1.0, y, lower = true)
            y
        }
        val trsv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.trsv(sparse, y, lower = true)
            y
        }

        assertTrue(gemv <= FLOOR_BYTES, "sparse gemv allocated $gemv B per call")
        assertTrue(symv <= FLOOR_BYTES, "sparse symv allocated $symv B per call")
        assertTrue(trsv <= FLOOR_BYTES, "sparse trsv allocated $trsv B per call")
    }

    @Test
    fun `sparse dense destinations reuse a workspace`() {
        val n = 64
        val rightHandSides = 8
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val identity = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val solveBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val multiplyBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val transposed = DenseMatrix.zero(rightHandSides, n)
        val square = DenseMatrix.zero(n, n)
        val panel = DenseMatrix.zero(n, rightHandSides)
        val workspace = Workspace()
        // One warm call per shape, so the loans exist before the measured loop asks for them again.
        engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
        engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
        engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
        engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
        engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
        engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
        engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)

        val left = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
            panel
        }
        val right = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
            transposed
        }
        val rank = bytesPerIteration(300, FLOOR_BYTES) {
            engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
            square
        }
        val symmetric = bytesPerIteration(300, FLOOR_BYTES) {
            engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
            panel
        }
        val sparseDense = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
            square
        }
        val solve = bytesPerIteration(300, FLOOR_BYTES) {
            engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
            solveBlock
        }
        val multiply = bytesPerIteration(300, FLOOR_BYTES) {
            engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)
            multiplyBlock
        }

        assertTrue(left <= FLOOR_BYTES, "left sparse gemm allocated $left B per call")
        assertTrue(right <= FLOOR_BYTES, "right transposed sparse gemm allocated $right B per call")
        assertTrue(rank <= FLOOR_BYTES, "sparse syrk allocated $rank B per call")
        assertTrue(symmetric <= FLOOR_BYTES, "sparse symm allocated $symmetric B per call")
        assertTrue(sparseDense <= FLOOR_BYTES, "direct sparse dense gemm allocated $sparseDense B per call")
        assertTrue(solve <= FLOOR_BYTES, "left sparse trsm allocated $solve B per call")
        assertTrue(multiply <= FLOOR_BYTES, "left sparse trmm allocated $multiply B per call")
    }
}
