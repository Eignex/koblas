package com.eignex.koblas

import kotlin.test.*

class RoutingTest {

    @Test
    fun `route queries reject negative sizes`() {
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemv(-1, 2) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseGemm(1, -1, 1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.DenseLu(-1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseDenseGemm(-1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseTriangularSolve(-1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseTriangularSolve(1, -1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseTriangularMultiply(-1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseTriangularMultiply(1, -1) }
        assertFailsWith<IllegalArgumentException> { F64RouteQuery.SparseLu(-1) }
    }

    @Test
    fun `displayed products saturate instead of overflowing`() {
        assertEquals(Long.MAX_VALUE, saturatedProduct(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
        assertEquals(24, saturatedProduct(2, 3, 4))
        assertEquals(0, saturatedProduct(0, Int.MAX_VALUE, Int.MAX_VALUE))
    }
}
