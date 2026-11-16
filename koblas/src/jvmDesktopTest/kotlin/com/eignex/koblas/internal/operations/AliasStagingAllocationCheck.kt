package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.testutil.allocation.*

/** Retained scratch must remove both snapshot-array allocation and temporary vector descriptors. */
internal object AliasStagingAllocationCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        val engine = KoblasEngine.select(args.single())
        val order = 17
        val matrix = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 0.01 })
        val values = DoubleArray(2 * order + 3)
        val vector = DenseVector.wrap(values, 2 * order + 1, order, -2)
        val scratch = Scratch()
        for (transpose in booleanArrayOf(false, true)) {
            assertAllocationFree("gemv aliased vectors transpose=$transpose") {
                values.fill(1.0)
                engine.gemv(0.5, matrix, vector, 0.25, vector, transpose, scratch)
                vector[0]
            }
        }
        for (lower in booleanArrayOf(false, true)) {
            assertAllocationFree("symv aliased vectors lower=$lower") {
                values.fill(1.0)
                engine.symv(0.5, matrix, vector, 0.25, vector, lower, scratch)
                vector[0]
            }
        }
        checkMatrixAliases(engine)
        val first = DenseVector.wrap(matrix.values, order - 1, order, -1)
        val second = DenseVector.wrap(matrix.values, order, order)
        val separate = DenseVector.wrap(DoubleArray(order) { 1.0 })
        for (aliases in 1..3) {
            val x = if (aliases != 2) first else separate
            val y = if (aliases != 1) second else separate
            assertAllocationFree("ger aliased operands=$aliases") {
                matrix.values.fill(0.01)
                engine.ger(0.5, x, y, matrix, scratch)
                matrix.values[0]
            }
        }
        assertAllocationFree("syr aliased operand") {
            matrix.values.fill(0.01)
            engine.syr(0.5, first, matrix, lower = true, scratch)
            matrix.values[0]
        }
        assertAllocationFree("syr2 identical aliases") {
            matrix.values.fill(0.01)
            engine.syr2(0.5, first, first, matrix, lower = true, scratch)
            matrix.values[0]
        }
        for (overlap in booleanArrayOf(false, true)) {
            val offset = if (overlap) order + 1 else 0
            assertAllocationFree("windowed ger overlapping=$overlap") {
                matrix.values.fill(0.01)
                engine.updateRank(
                    0.5, matrix.values, offset, 1, separate.values, 0, 1, matrix,
                    1, 1, order - 1, order - 1, scratch,
                )
                matrix.values[order + 1]
            }
        }
    }

    private fun checkMatrixAliases(engine: KoblasEngine) {
        val a = DenseMatrix.wrap(9, 9, DoubleArray(13 * 12 + 5), 2, 13)
        val destination = DenseVector.wrap(a.values, a.offset + 8, 9, -1)
        val source = DenseVector.wrap(DoubleArray(9) { 1.0 })
        val scratch = Scratch()
        for (transpose in booleanArrayOf(false, true)) {
            assertAllocationFree("gemv aliased matrix transpose=$transpose") {
                a.values.fill(0.01)
                engine.gemv(0.5, a, source, 0.25, destination, transpose, scratch)
                destination[0]
            }
        }
        for (lower in booleanArrayOf(false, true)) {
            assertAllocationFree("symv aliased matrix lower=$lower") {
                a.values.fill(0.01)
                engine.symv(0.5, a, source, 0.25, destination, lower, scratch)
                destination[0]
            }
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    for (solve in booleanArrayOf(false, true)) {
                        val name = "triangular alias lower=$lower transpose=$transpose unit=$unit solve=$solve"
                        assertAllocationFree(name) {
                            a.values.fill(0.01)
                            for (i in 0 until a.rows) a[i, i] = 4.0
                            if (solve) {
                                engine.trsv(a, destination, lower, transpose, unit, scratch)
                            } else {
                                engine.trmv(a, destination, lower, transpose, unit, scratch)
                            }
                            destination[0]
                        }
                    }
                }
            }
        }
    }

    private fun assertAllocationFree(name: String, probe: AllocationProbe) {
        val bytes = bytesPerCall(probe, warmup = 20_000, iterations = 2_000, windows = 8)
        check(bytes <= 8.0) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }
}
