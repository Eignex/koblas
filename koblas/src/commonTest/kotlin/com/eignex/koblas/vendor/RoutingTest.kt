package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.test.Test
import kotlin.test.assertEquals

class RoutingTest {
    @Test
    fun `zero depth products still report the destination scaling call`() {
        val empty = DenseMatrix.zero(3, 0)
        val destination = DenseMatrix.zero(3)
        for (operation in listOf(BlasOperation.Gemm, BlasOperation.Gemmt, BlasOperation.Syrk, BlasOperation.Syr2k)) {
            val matrices = when (operation) {
                BlasOperation.Gemm, BlasOperation.Gemmt -> listOf(empty, DenseMatrix.zero(0, 3), destination)
                BlasOperation.Syrk -> listOf(empty, destination)
                else -> listOf(empty, empty, destination)
            }
            for (exported in booleanArrayOf(true, false)) {
                val route = routeFor(operation, Vendor.OpenBlas, exported, matrices, emptyList(), null)

                assertEquals(if (exported) RouteKind.Direct else RouteKind.Composed, route.kind, operation.name)
            }
        }
    }

    @Test
    fun `empty destinations still report no work for matrix routines`() {
        for (operation in BlasOperation.entries.filter { it.level == 3 }) {
            val matrices = if (operation == BlasOperation.Syrk || operation == BlasOperation.Trmm ||
                operation == BlasOperation.Trsm
            ) {
                listOf(DenseMatrix.zero(0), DenseMatrix.zero(0))
            } else {
                List(3) { DenseMatrix.zero(0) }
            }

            val route = routeFor(operation, Vendor.OpenBlas, true, matrices, emptyList(), null)

            assertEquals(RouteKind.NoWork, route.kind, operation.name)
        }
    }

    @Test
    fun `empty matrix vector operands retain the gemv quick return`() {
        val route = routeFor(
            BlasOperation.Gemv,
            Vendor.OpenBlas,
            true,
            listOf(DenseMatrix.zero(3, 0)),
            listOf(DenseVector.zero(0), DenseVector.zero(3)),
            null,
        )

        assertEquals(RouteKind.NoWork, route.kind)
    }
}
