package com.eignex.koblas

import kotlin.test.*

class RoutingTest {
    @Test
    fun `route queries reject negative sizes`() {
        assertFailsWith<IllegalArgumentException> { RouteQuery.DenseGemv(-1, 2) }
        assertFailsWith<IllegalArgumentException> { RouteQuery.DenseGemm(1, -1, 1) }
        assertFailsWith<IllegalArgumentException> { RouteQuery.SparseDenseGemm(-1) }
        assertFailsWith<IllegalArgumentException> {
            RouteQuery.SparseTriangular(-1, kind = SparseTriangularKind.SOLVE)
        }
        assertFailsWith<IllegalArgumentException> {
            RouteQuery.SparseTriangular(1, kind = SparseTriangularKind.SOLVE, rightHandSides = -1)
        }
        assertFailsWith<IllegalArgumentException> {
            RouteQuery.SparseTriangular(-1, kind = SparseTriangularKind.MULTIPLY)
        }
        assertFailsWith<IllegalArgumentException> {
            RouteQuery.SparseTriangular(1, kind = SparseTriangularKind.MULTIPLY, rightHandSides = -1)
        }
        assertFailsWith<IllegalArgumentException> { RouteQuery.SparseLu(-1) }
    }
}
