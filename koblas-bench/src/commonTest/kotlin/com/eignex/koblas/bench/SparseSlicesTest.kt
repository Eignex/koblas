package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.SparseOperation
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SparseSlicesTest {
    @Test
    fun `slice cases use each explicit engine and include validation`() {
        for (operation in listOf("spdot-slice", "spnrm2-slice", "spaxpy-slice", "spscatter-slice")) {
            val case = Cases.parse("$operation+37+sparse-uniform+density=0.5").single()
            val selected = when (operation) {
                "spdot-slice" -> SparseOperation.DotDense
                "spnrm2-slice" -> SparseOperation.IndexedNrm2
                "spaxpy-slice" -> SparseOperation.Axpy
                else -> SparseOperation.Scatter
            }
            for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
                val arm = assertNotNull(sparseArm(case, engine))
                val work = assertNotNull(arm.work)
                val count = Fixtures.sparseVector(37, 0.5, 1).nnz
                assertEquals(engine.routeOf(selected, count).kind.name.lowercase(), work.comparisonKind)
                assertContains(assertNotNull(work.kernel), "portable-index-validation+")
                val reduction = operation == "spdot-slice" || operation == "spnrm2-slice"
                assertEquals(if (reduction) "arithmetic" else "reset-and-arithmetic", work.timingMode)
                val first = work.run()
                assertTrue(first.isFinite())
                assertEquals(first, work.run())
                if (!reduction) assertNotNull(work.result)
            }
        }
    }
}
