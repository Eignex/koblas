package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Each probe owns its fixture so mutations and workspace loans cannot leak between cases.
 * Portable Level Two probes cover scheduling; instrumentation prevents Vector API carrier replacement,
 * so the uninstrumented SIMD allocation tasks measure the vector panels separately.
 */
@RunWith(Parameterized::class)
class AllocationFreeTest(
    private val name: String,
    private val iterations: Int,
    private val probe: () -> Any?,
) {
    @Test
    fun `warmed calls stay within their allocation budget`() {
        val bytes = bytesPerIteration(iterations, FLOOR_BYTES, block = probe)

        assertTrue(bytes <= FLOOR_BYTES, "$name allocated $bytes B per call")
    }

    companion object {
        private const val FLOOR_BYTES = 64.0
        private val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun probes(): Collection<Array<Any>> = listOf(
            arrayOf<Any>("disjoint shared buffer vector copy allocates nothing", 1_000, with(SharedBufferFixture()) { {
                copy(source, destination)
                backing
            } }),
            arrayOf<Any>("disjoint shared buffer vector axpy allocates nothing", 1_000, with(SharedBufferFixture()) { {
                destination.axpy(1e-12, source)
                backing
            } }),
            arrayOf<Any>("disjoint shared buffer vector swap allocates nothing", 1_000, with(SharedBufferFixture()) { {
                swap(source, destination)
                backing
            } }),
            arrayOf<Any>("dense dot allocates nothing", 1_000, with(DenseVectorFixture()) { {
                kernels.dot(a, 0, b, 0, n)
            } }),
            arrayOf<Any>("dense axpy allocates nothing", 1_000, with(DenseVectorFixture()) { {
                kernels.axpy(b, 0, 0.5, a, 0, n)
            } }),
            arrayOf<Any>("dense norm allocates nothing", 1_000, with(DenseVectorFixture()) { {
                kernels.nrm2(a, 0, n)
            } }),
            arrayOf<Any>("strided dense dot allocates nothing", 1_000, with(DenseVectorFixture()) { {
                kernels.dot(a, 0, b, 0, n / 2, 2, 2)
            } }),
            arrayOf<Any>("dense gemv allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.gemv(1e-12, a, x, 1.0, y)
                y
            } }),
            arrayOf<Any>("transposed dense gemv allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.gemv(1e-12, a, x, 1.0, y, transpose = true)
                y
            } }),
            arrayOf<Any>("dense symv allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.symv(1e-12, a, x, 1.0, y)
                y
            } }),
            arrayOf<Any>("dense ger allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.ger(1e-12, x, y, a)
                a
            } }),
            arrayOf<Any>("dense syr allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.syr(1e-12, view, a)
                a
            } }),
            arrayOf<Any>("dense trmv allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.trmv(triangle, y, lower = true)
                y
            } }),
            arrayOf<Any>("dense trsv allocates nothing", 400, with(DenseLevelTwoFixture()) { {
                portable.trsv(triangle, y, lower = true)
                y
            } }),
            arrayOf<Any>("contiguous gemv allocates nothing", 1_000, with(ContiguousFixture()) { {
                a.gemvInto(1e-12, contiguous, 1.0, destination)
                destination
            } }),
            arrayOf<Any>("contiguous symv allocates nothing", 1_000, with(ContiguousFixture()) { {
                a.symvInto(1e-12, contiguous, 1.0, destination)
                destination
            } }),
            arrayOf<Any>("aliased dense gemv reuses a workspace", 400, with(AliasedLevelTwoFixture()) { {
                portable.gemv(1e-12, a, y, 1.0, y, workspace = workspace)
                y
            } }),
            arrayOf<Any>("aliased dense symv reuses a workspace", 400, with(AliasedLevelTwoFixture()) { {
                portable.symv(1e-12, a, y, 1.0, y, workspace = workspace)
                y
            } }),
            arrayOf<Any>("aliased sparse gemv reuses a workspace", 400, with(AliasedLevelTwoFixture()) { {
                koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, workspace = workspace)
                coefficients
            } }),
            arrayOf<Any>("dense trsm reuses a workspace", 100, with(DenseLevelThreeFixture()) { {
                portable.trsm(triangle, b, lower = true, workspace = workspace)
                b
            } }),
            arrayOf<Any>("dense trmm reuses a workspace", 100, with(DenseLevelThreeFixture()) { {
                portable.trmm(triangle, b, lower = true, workspace = workspace)
                b
            } }),
            arrayOf<Any>("shallow matrix products reuse a workspace", 50, with(DenseProductFixture()) { {
                portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
                packedTarget
            } }),
            arrayOf<Any>("deep matrix products reuse a workspace", 50, with(DenseProductFixture()) { {
                portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)
                deepTarget
            } }),
            arrayOf<Any>("retained matrix products allocate nothing", 50, with(DenseProductFixture()) { {
                portable.gemm(1e-12, left, right, 1.0, packedTarget)
                packedTarget
            } }),
            arrayOf<Any>("unpacked matrix products reuse a workspace", 200, with(DenseProductFixture()) { {
                portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
                smallTarget
            } }),
            arrayOf<Any>("sparse gemv allocates nothing", 1_000, with(SparseLevelTwoFixture()) { {
                engine.gemv(1e-12, sparse, x, 1.0, y, transpose = true)
                y
            } }),
            arrayOf<Any>("sparse symv allocates nothing", 1_000, with(SparseLevelTwoFixture()) { {
                engine.symv(1e-12, sparse, x, 1.0, y, lower = true)
                y
            } }),
            arrayOf<Any>("sparse trsv allocates nothing", 1_000, with(SparseLevelTwoFixture()) { {
                engine.trsv(sparse, y, lower = true)
                y
            } }),
            arrayOf<Any>("left sparse gemm reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
                panel
            } }),
            arrayOf<Any>("right sparse gemm reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
                transposed
            } }),
            arrayOf<Any>("sparse syrk reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
                square
            } }),
            arrayOf<Any>("sparse symm reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
                panel
            } }),
            arrayOf<Any>("sparse matrix product reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
                square
            } }),
            arrayOf<Any>("sparse trsm reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
                solveBlock
            } }),
            arrayOf<Any>("sparse trmm reuses a workspace", 300, with(SparseLevelThreeFixture()) { {
                engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)
                multiplyBlock
            } }),
        )
    }

    private class SharedBufferFixture {
        val n = 512
        val backing = DoubleArray(2 * n) { it * 0.01 }
        val source = StridedVector(backing, 0, n, 2)
        val destination = StridedVector(backing, 1, n, 2)
    }

    private class DenseVectorFixture {
        val n = 512
        val a = DoubleArray(n) { it * 0.01 }
        val b = DoubleArray(n) { 1.0 / (it + 1) }
        val kernels = engine.vectorKernels
    }

    private class DenseLevelTwoFixture {
        val n = 48
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 0.25 + (it % 11) * 0.0625 }).also {
            for (i in 0 until n) it.values[i + i * n] = 4.0
        }
        val x = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(n) { 0.5 }
        val view = DenseVector.wrap(x)
        val portable = BuiltinEngines.scalar
    }

    private class ContiguousFixture {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val contiguous = DenseVector.wrap(DoubleArray(n) { 1.0 + (it % 7) * 0.25 })
        val destination = DoubleArray(n)
    }

    private class AliasedLevelTwoFixture {
        val n = 128
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val y = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf((j + 1) % n to 1.0 + j % 5) })
        val coefficients = sparse.values
        val workspace = Workspace()
        init {
            portable.gemv(1e-12, a, y, 1.0, y, workspace = workspace)
            koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, workspace = workspace)
        }
    }

    private class DenseLevelThreeFixture {
        val order = 32
        val sides = 8
        val portable = BuiltinEngines.scalar
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 0.25 + (it % 11) * 0.0625 }).also {
            for (i in 0 until order) it.values[i + i * order] = 4.0
        }
        val b = DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { 1.0 })
        val workspace = Workspace()
        init {
            portable.trsm(triangle, b, lower = true, workspace = workspace)
            portable.trmm(triangle, b, lower = true, workspace = workspace)
        }
    }

    private class DenseProductFixture {
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
        init {
            portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
            portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)
            portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
        }
    }

    private class SparseLevelTwoFixture {
        val n = 128
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val x = DoubleArray(n) { it * 0.01 }
        val y = DoubleArray(n)
    }

    private class SparseLevelThreeFixture {
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
        init {
            // One warm call per shape, so the loans exist before the measured loop asks for them again.
            engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
            engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
            engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
            engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
            engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
            engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
            engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)
        }
    }
}
