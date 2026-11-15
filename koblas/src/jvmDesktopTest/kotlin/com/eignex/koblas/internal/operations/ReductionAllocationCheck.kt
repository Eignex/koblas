package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.testutil.allocation.*

/** Reduction adapters must not need the JIT to inline view factories to avoid temporary containers. */
internal object ReductionAllocationCheck {
    @Volatile
    private var resultSink: DoubleArray? = null

    @JvmStatic
    fun main(args: Array<String>) {
        val engine = KoblasEngine.select(args.single())
        val packed = DenseMatrix.wrap(17, 21, DoubleArray(17 * 21) { (it % 17 - 8) / 16.0 })
        val padded = DenseMatrix.wrap(17, 21, DoubleArray(40 * 21 + 3) { (it % 17 - 8) / 16.0 }, 3, 40)
        for (matrix in listOf(packed, padded)) {
            val name = if (matrix === packed) "packed" else "padded"
            assertAllocationFree("$name sum") { engine.sumMatrix(matrix) }
            for (type in listOf(Norm.One, Norm.Infinity, Norm.Frobenius, Norm.MaxAbs)) {
                assertAllocationFree("$name norm $type") { engine.normMatrix(matrix, type) }
            }
            checkAxisSums(engine, matrix, name)
        }
        val sparse = SparseMatrix.wrap(
            17,
            4,
            intArrayOf(0, 2, 2, 3, 4),
            intArrayOf(1, 16, 4, 2),
            doubleArrayOf(-2.0, 3.0, -4.0, 5.0),
        )
        assertAllocationFree("sparse sum") { engine.sumMatrix(sparse) }
        for (type in listOf(Norm.One, Norm.Frobenius, Norm.MaxAbs)) {
            assertAllocationFree("sparse norm $type") { engine.normMatrix(sparse, type) }
        }
        checkAxisSums(engine, sparse, "sparse", axis = 0)
        val vector = SparseVector.wrap(17, intArrayOf(1, 16), doubleArrayOf(-2.0, 3.0))
        assertAllocationFree("sparse vector maximum magnitude") { engine.normVector(vector, Norm.MaxAbs) }
    }

    private fun checkAxisSums(engine: KoblasEngine, matrix: Matrix, name: String, axis: Int? = null) {
        for (selected in if (axis == null) intArrayOf(0, 1) else intArrayOf(axis)) {
            val size = if (selected == 0) matrix.cols else matrix.rows
            val resultBytes = ARRAY_HEADER_BYTES + size * Double.SIZE_BYTES
            assertAllocationFree("$name axis $selected sum", resultBytes.toDouble()) {
                val result = engine.sumMatrix(matrix, selected)
                resultSink = result
                result[0]
            }
        }
    }

    private fun assertAllocationFree(name: String, resultBytes: Double = 0.0, probe: AllocationProbe) {
        val bytes = bytesPerCall(probe, warmup = 20_000, iterations = 2_000, windows = 8)
        check(bytes <= resultBytes + 8.0) { "$name allocated $bytes B per call including $resultBytes B of results" }
        println("$name allocated $bytes B per call including $resultBytes B of results")
    }
}
