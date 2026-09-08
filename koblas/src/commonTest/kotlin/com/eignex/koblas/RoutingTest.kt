package com.eignex.koblas

import kotlin.test.*

class RoutingTest {
    @Test
    fun `route queries reject negative sizes`() {
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemv(-1, 2) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemm(1, -1, 1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseDenseGemm(-1) }
        assertFailsWith<IllegalArgumentException> {
            F64RouteQuery.SparseTriangular(-1, kind = SparseTriangularKind.SOLVE)
        }
        assertFailsWith<IllegalArgumentException> {
            F64RouteQuery.SparseTriangular(1, kind = SparseTriangularKind.SOLVE, rightHandSides = -1)
        }
        assertFailsWith<IllegalArgumentException> {
            F64RouteQuery.SparseTriangular(-1, kind = SparseTriangularKind.MULTIPLY)
        }
        assertFailsWith<IllegalArgumentException> {
            F64RouteQuery.SparseTriangular(1, kind = SparseTriangularKind.MULTIPLY, rightHandSides = -1)
        }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseLu(-1) }
    }
}
