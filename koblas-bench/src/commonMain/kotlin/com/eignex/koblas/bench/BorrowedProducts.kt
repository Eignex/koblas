package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseMatrixRegion
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.routeOf

/** An addressed product, including reset and checks over every borrowed backing-buffer entry. */
internal fun borrowedProductWork(case: BenchCase, engine: KoblasEngine): CaseWork {
    val (rows, columns, depth) = case.dimensions
    val ta = case.flag("transA")
    val tb = case.flag("transB")
    val alpha = 0.875
    val beta = -0.25
    val sourceA = Fixtures.matrix(if (ta) depth else rows, if (ta) rows else depth, 1)
    val sourceB = Fixtures.matrix(if (tb) columns else depth, if (tb) depth else columns, 2)
    val initialC = Fixtures.matrix(rows, columns, 3)
    fun borrow(source: DenseMatrix): DenseMatrixRegion {
        val parent = DenseMatrix.wrap(source.rows + 3, source.cols + 2,
            DoubleArray((source.rows + 3) * (source.cols + 2)) { -77.0 })
        val region = parent.view(1, 1, source.rows, source.cols)
        for (j in 0 until source.cols) for (i in 0 until source.rows) region[i, j] = source[i, j]
        return region
    }
    val a = borrow(sourceA)
    val b = borrow(sourceB)
    val c = borrow(initialC)
    val original = c.values.copyOf()
    val expected = original.copyOf()
    val result = DenseReference.gemm(alpha, sourceA, ta, sourceB, tb, beta, initialC)
    for (j in 0 until columns) for (i in 0 until rows) {
        expected[c.offset + i + j * c.leadingDimension] = result[i + j * rows]
    }
    val workspace = if (case.options["workspace"] == "retained") Workspace() else null
    fun apply(): Double {
        original.copyInto(c.values)
        engine.gemm(alpha, a, ta, b, tb, beta, c, workspace)
        return c[0, 0]
    }
    apply()
    DenseReference.check(expected, c.values, "${case.id} borrowed backing")
    val route = engine.routeOf(alpha, a, ta, b, tb, beta, c)
    return CaseWork(route.kind.name.lowercase(), "reset-and-arithmetic", ::apply, kernel = matrixKernel(route))
}
