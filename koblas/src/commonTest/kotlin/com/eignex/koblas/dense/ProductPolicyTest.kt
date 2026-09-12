package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals

class ProductPolicyTest {
    @Test
    fun `conservative products stay direct unless operands are retained`() {
        val policy = ProductPolicy()
        assertEquals(ProductPacking.Direct, policy.packing(32, 32, 32, false, false))
        assertEquals(ProductPacking.Left, policy.packing(32, 32, 32, true, false))
        assertEquals(ProductPacking.Right, policy.packing(32, 32, 32, false, true))
        assertEquals(ProductPacking.Both, policy.packing(32, 32, 32, true, true))
    }

    @Test
    fun `shape rules select packing before preparing data`() {
        val policy = ProductPolicy(
            packLeft = WorkRule.Shape(4, 16, 4, 64),
            packRight = WorkRule.Shape(16, 4, 4, 64),
            packBoth = WorkRule.Shape(16, 16, 65, 128),
        )
        assertEquals(ProductPacking.Direct, policy.packing(1, 64, 8, false, false))
        assertEquals(ProductPacking.Left, policy.packing(4, 64, 8, false, false))
        assertEquals(ProductPacking.Right, policy.packing(64, 4, 8, false, false))
        assertEquals(ProductPacking.Both, policy.packing(32, 32, 100, false, false))
    }
}
