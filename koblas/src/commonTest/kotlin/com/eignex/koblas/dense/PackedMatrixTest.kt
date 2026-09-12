package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PackedMatrixTest {
    @Test
    fun `borrowed packed storage remains live`() {
        val data = DoubleArray(9)
        val layout = PackedMatrixLayout(PackedRole.Left, 2, 2, 2)
        val packed = PackedMatrix.wrap(data, layout, offset = 3)
        data[3] = 7.0
        assertEquals(7.0, packed[0, 0])
        assertEquals(PackedOwnership.Borrowed, packed.ownership)
    }

    @Test
    fun `invalid physical descriptors fail before wrapping`() {
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, Int.MAX_VALUE, 2, 2) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, Int.MAX_VALUE, 2) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, 2, 2, depthStride = 1) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, 2, 2, groupStride = 1) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, 2, 2, version = 2) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, 2, 2, layoutId = 99) }
        assertFailsWith<IllegalArgumentException> { PackedMatrixLayout(PackedRole.Left, 2, 2, 2, alignmentBytes = 64) }
        assertFailsWith<IllegalArgumentException> {
            PackedMatrix.wrap(DoubleArray(4), PackedMatrixLayout(PackedRole.Left, 2, 2, 2), offset = 1)
        }
    }

    @Test
    fun `streaming restriction belongs to the retained layout`() {
        val layout = PackedMatrixLayout(PackedRole.Left, 2, 2, 2, requiredSvlBytes = 64)
        layout.requireCompatible(PackedMatrixLayout.GROUPED_FP64, 1, 64)
        assertFailsWith<IllegalArgumentException> { layout.requireCompatible(PackedMatrixLayout.GROUPED_FP64, 1, 32) }
        assertFailsWith<IllegalArgumentException> { layout.requireCompatible(PackedMatrixLayout.GROUPED_FP64, 1) }
        assertFailsWith<IllegalArgumentException> { layout.requireCompatible(PackedMatrixLayout.GROUPED_FP64, 2, 64) }
    }
}
