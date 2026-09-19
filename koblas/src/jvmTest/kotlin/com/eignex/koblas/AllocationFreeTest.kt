package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The seams a caller may run inside a hot loop without the collector noticing.
 *
 * Level 1 and the sparse primitives take raw arrays and offsets precisely so a caller sweeping a matrix can
 * reach them per sub-range without wrapping anything, and this is what holds them to it. Level 2 and 3 make no
 * such promise on their own. The sparse matrix algorithms do, given a workspace: their staging, accumulator
 * and panel scratch are loans, so a repeated call over the same shapes reuses them instead of allocating. An
 * operation that discovers a new structure, such as a fresh CSC product, allocates its result by definition
 * and is not measured here.
 */
class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

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
        val workspace = MatrixWorkspace()
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
