package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.testutil.allocation.*

/** Retained matrix snapshots must not depend on inlining a container factory to avoid allocation. */
internal object MatrixAliasAllocationCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        val engine = KoblasEngine.select(args.single())
        val a = DenseMatrix.wrap(9, 9, DoubleArray(13 * 12 + 5), 2, 13)
        val shared = a.view()
        val overlap = DenseMatrix.wrap(9, 9, a.values, 3, 13)
        val b = DenseMatrix.wrap(9, 9, DoubleArray(81) { 0.01 })
        val scratch = Scratch()
        for (other in listOf(b, shared, overlap)) {
            val alias = when {
                other === b -> "left"
                other === shared -> "identical"
                else -> "both"
            }
            assertAllocationFree("gemm aliases=$alias") {
                a.values.fill(0.01)
                engine.gemm(0.5, a, false, other, false, 0.25, a, scratch)
                a.values[a.offset]
            }
        }
        for (lower in booleanArrayOf(false, true)) {
            assertAllocationFree("gemmt aliases lower=$lower") {
                a.values.fill(0.01)
                engine.gemmt(0.5, a, false, overlap, false, 0.25, a, lower, scratch)
                a.values[a.offset]
            }
            for (right in booleanArrayOf(false, true)) {
                assertAllocationFree("symm shared aliases lower=$lower right=$right") {
                    a.values.fill(0.01)
                    engine.symm(0.5, a, shared, 0.25, a, lower, right, scratch)
                    a.values[a.offset]
                }
            }
        }
        for (transpose in booleanArrayOf(false, true)) {
            assertAllocationFree("syrk alias transpose=$transpose") {
                a.values.fill(0.01)
                engine.syrk(0.5, a, transpose, 0.25, a, lower = true, scratch)
                a.values[a.offset]
            }
            assertAllocationFree("syr2k aliases transpose=$transpose") {
                a.values.fill(0.01)
                engine.syr2k(0.5, a, overlap, transpose, 0.25, a, lower = true, scratch)
                a.values[a.offset]
            }
        }
        checkTriangles(engine, a, scratch)
        checkThinProducts(engine)
        val wide = DenseMatrix.wrap(49, 49, DoubleArray(49 * 49))
        val wideOther = DenseMatrix.wrap(49, 49, DoubleArray(49 * 49) { 0.01 })
        assertAllocationFree("wide gemm alias", warmup = 2_000, iterations = 200) {
            wide.values.fill(0.01)
            engine.gemm(0.5, wide, false, wideOther, false, 0.25, wide, scratch)
            wide.values[0]
        }
    }

    private fun checkTriangles(engine: KoblasEngine, a: DenseMatrix, scratch: Scratch) {
        for (right in booleanArrayOf(false, true)) {
            for (unit in booleanArrayOf(false, true)) {
                for (solve in booleanArrayOf(false, true)) {
                    assertAllocationFree("triangular alias right=$right unit=$unit solve=$solve") {
                        a.values.fill(0.01)
                        for (i in 0 until a.rows) a[i, i] = 4.0
                        if (solve) {
                            engine.trsm(a, a, lower = true, unitDiag = unit, right = right, scratch = scratch)
                        } else {
                            engine.trmm(a, a, lower = true, unitDiag = unit, right = right, scratch = scratch)
                        }
                        a.values[a.offset]
                    }
                }
            }
        }
    }

    private fun checkThinProducts(engine: KoblasEngine) {
        val parent = DenseMatrix.wrap(12, 12, DoubleArray(12 * 12))
        val scratch = Scratch()
        for (right in booleanArrayOf(false, true)) {
            val rows = if (right) 1 else 5
            val columns = if (right) 5 else 1
            val c = parent.view(1, 1, rows, columns)
            for (ta in booleanArrayOf(false, true)) {
                for (tb in booleanArrayOf(false, true)) {
                    val a = parent.view(1, 1, if (ta) 3 else rows, if (ta) rows else 3)
                    val b = parent.view(1, 1, if (tb) columns else 3, if (tb) 3 else columns)
                    assertAllocationFree("thin gemm aliases right=$right ta=$ta tb=$tb") {
                        parent.values.fill(0.01)
                        engine.gemm(0.5, a, ta, b, tb, 0.25, c, scratch)
                        parent.values[c.offset]
                    }
                }
            }
        }
    }

    private fun assertAllocationFree(
        name: String,
        warmup: Int = 20_000,
        iterations: Int = 2_000,
        probe: AllocationProbe,
    ) {
        val bytes = bytesPerCall(probe, warmup, iterations, windows = 8)
        check(bytes <= 8.0) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }
}
