package com.eignex.koblas

import kotlin.test.*

class RoutingTest {

    @Test
    fun `route queries reject negative sizes`() {
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemv(-1, 2) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemm(1, -1, 1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseLu(-1) }
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
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.Level1(F64Level1Routine.DOT, -1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.CholeskyRankUpdate(order = -1, rank = 1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.CholeskyRankUpdate(order = 1, rank = -1) }
    }

    @Test
    fun `displayed products saturate instead of overflowing`() {
        assertEquals(Long.MAX_VALUE, saturatedProduct(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
        assertEquals(24, saturatedProduct(2, 3, 4))
        assertEquals(0, saturatedProduct(0, Int.MAX_VALUE, Int.MAX_VALUE))
    }
}
