package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.testutil.allocation.*

/** Sparse snapshots retain only coefficient arrays, including when every operand shares the destination. */
internal object SparseAliasAllocationCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        val engine = KoblasEngine.select(args.single())
        val order = 5
        val values = DoubleArray(order * order)
        val a = SparseMatrix.wrap(
            order,
            order,
            IntArray(order + 1) { it * order },
            IntArray(values.size) { it % order },
            values,
        )
        val b = DenseMatrix.wrap(order, order, values)
        val scratch = Scratch()
        for (right in booleanArrayOf(false, true)) {
            for (ta in booleanArrayOf(false, true)) {
                for (tb in booleanArrayOf(false, true)) {
                    assertAllocationFree("sparse gemm all aliases right=$right ta=$ta tb=$tb") {
                        values.fill(0.01)
                        engine.gemm(0.5, a, ta, b, tb, 0.25, b, right, scratch)
                        values[0]
                    }
                }
            }
            for (lower in booleanArrayOf(false, true)) {
                assertAllocationFree("sparse symm all aliases right=$right lower=$lower") {
                    values.fill(0.01)
                    engine.symm(0.5, a, b, 0.25, b, lower, right, scratch)
                    values[0]
                }
            }
        }
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                assertAllocationFree("sparse syrk alias transpose=$transpose lower=$lower") {
                    values.fill(0.01)
                    engine.syrk(0.5, a, transpose, 0.25, b, lower, scratch)
                    values[0]
                }
            }
        }
        checkTriangles(engine, a, b, scratch)
        checkVectorAliases(engine)
        checkPaddedAliases(engine, a, scratch)
    }

    private fun checkTriangles(engine: KoblasEngine, a: SparseMatrix, b: DenseMatrix, scratch: Scratch) {
        for (right in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unit in booleanArrayOf(false, true)) {
                        for (solve in booleanArrayOf(false, true)) {
                            val name = "sparse triangular alias right=$right lower=$lower trans=$transpose" +
                                " unit=$unit solve=$solve"
                            assertAllocationFree(name) {
                                a.values.fill(0.01)
                                for (i in 0 until a.rows) b[i, i] = 4.0
                                if (solve) {
                                    engine.trsm(a, b, lower, transpose, unit, right, 0.5, scratch)
                                } else {
                                    engine.trmm(a, b, lower, transpose, unit, right, 0.5, scratch)
                                }
                                a.values[0]
                            }
                        }
                    }
                }
            }
        }
    }

    private fun checkVectorAliases(engine: KoblasEngine) {
        val order = 5
        val values = DoubleArray(order)
        val a = SparseMatrix.wrap(order, order, IntArray(order + 1) { it }, IntArray(order) { it }, values)
        val vector = DenseVector.wrap(values)
        val scratch = Scratch()
        for (flag in booleanArrayOf(false, true)) {
            assertAllocationFree("sparse gemv all aliases transpose=$flag") {
                values.fill(1.0)
                engine.gemv(0.5, a, vector, 0.25, vector, flag, scratch)
                values[0]
            }
            assertAllocationFree("sparse symv all aliases lower=$flag") {
                values.fill(1.0)
                engine.symv(0.5, a, vector, 0.25, vector, flag, scratch)
                values[0]
            }
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    for (solve in booleanArrayOf(false, true)) {
                        val name = "sparse triangular vector alias lower=$flag trans=$transpose unit=$unit solve=$solve"
                        assertAllocationFree(name) {
                            values.fill(1.0)
                            if (solve) {
                                engine.trsv(a, vector, flag, transpose, unit, scratch)
                            } else {
                                engine.trmv(a, vector, flag, transpose, unit, scratch)
                            }
                            values[0]
                        }
                    }
                }
            }
        }
    }

    private fun checkPaddedAliases(engine: KoblasEngine, a: SparseMatrix, scratch: Scratch) {
        for (right in booleanArrayOf(false, true)) {
            val b = DenseMatrix.wrap(if (right) 3 else 5, if (right) 5 else 3, a.values, 1, if (right) 5 else 6)
            assertAllocationFree("sparse gemm padded aliases right=$right") {
                a.values.fill(0.01)
                engine.gemm(0.5, a, false, b, false, 0.25, b, right, scratch)
                a.values[1]
            }
        }
    }

    private fun assertAllocationFree(name: String, probe: AllocationProbe) {
        val bytes = bytesPerCall(probe, warmup = 20_000, iterations = 2_000, windows = 8)
        check(bytes <= 8.0) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }
}
