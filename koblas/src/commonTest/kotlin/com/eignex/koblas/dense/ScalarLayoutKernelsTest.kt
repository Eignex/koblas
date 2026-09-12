package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ScalarLayoutKernelsTest {
    @Test
    fun `rectangular triangular packing preserves implicit zeros when scaled`() {
        val source = MatrixWindow(DoubleArray(25) { Double.NaN }, 5, 5, structure = MatrixStructure.UnitLower)
            .window(1, 2, 2, 3)
        val packed = ScalarLayoutKernels.pack(
            source,
            PackedMatrixLayout(PackedRole.Left, 2, 3, 3),
            Double.POSITIVE_INFINITY,
        )

        for (j in 0 until 3) {
            for (i in 0 until 2) {
                if (i == 1 && j == 0) {
                    assertEquals(Double.POSITIVE_INFINITY, packed[i, j])
                } else {
                    assertEquals(0L, packed[i, j].toRawBits())
                }
            }
        }
    }

    @Test
    fun `packing round trips preserve bits and clear only physical padding`() {
        val values = doubleArrayOf(-0.0, Double.fromBits(0x7ff8000000000123L), 3.0, 4.0, 5.0, 6.0)
        val source = MatrixWindow(values, 3, 2)
        for (role in PackedRole.entries) {
            for (group in listOf(1, 2, 4)) {
                val depth = if (role == PackedRole.Left) 2 else 3
                val layout = PackedMatrixLayout(
                    role,
                    3,
                    2,
                    group,
                    depthStride = group + 1,
                    groupStride = depth * (group + 1) + 2,
                )
                val backing = DoubleArray(layout.storageSize + 6) { -7.0 }

                val packed = ScalarLayoutKernels.packInto(source, backing, layout, offset = 3)
                val result = DoubleArray(11) { -8.0 }
                ScalarLayoutKernels.unpack(packed, MatrixWindow(result, 3, 2, offset = 1, columnStride = 5))

                val logical = mutableSetOf<Int>()
                for (j in 0 until 2) {
                    for (i in 0 until 3) {
                        assertEquals(values[i + j * 3].toRawBits(), result[1 + i + j * 5].toRawBits())
                        logical += 3 + layout.index(i, j)
                    }
                }
                for (index in backing.indices) {
                    if (index !in logical) {
                        assertEquals(
                            if (index in 3 until 3 + layout.storageSize) 0.0 else -7.0,
                            backing[index],
                        )
                    }
                }
                for (index in listOf(0, 4, 5, 9, 10)) assertEquals(-8.0, result[index])
            }
        }
    }

    @Test
    fun `structured pack and transpose never propagate ignored storage`() {
        for (structure in MatrixStructure.entries.filter { it != MatrixStructure.General }) {
            val lower = structure in listOf(
                MatrixStructure.SymmetricLower,
                MatrixStructure.TriangularLower,
                MatrixStructure.UnitLower,
            )
            val unit = structure in listOf(MatrixStructure.UnitLower, MatrixStructure.UnitUpper)
            val data = DoubleArray(9) { index ->
                val i = index % 3
                val j = index / 3
                if ((if (lower) i >= j else i <= j) && !(unit && i == j)) (index + 1).toDouble() else Double.NaN
            }
            val source = MatrixWindow(data, 3, 3, structure = structure)
            for (role in PackedRole.entries) {
                val packed = ScalarLayoutKernels.pack(source, PackedMatrixLayout(role, 3, 3, 2))
                for (j in 0 until 3) {
                    for (i in 0 until 3) {
                        val expected = when {
                            unit && i == j -> 1.0

                            (if (lower) i >= j else i <= j) -> data[i + j * 3]

                            structure in listOf(
                                MatrixStructure.SymmetricLower,
                                MatrixStructure.SymmetricUpper,
                            ) -> data[j + i * 3]

                            else -> 0.0
                        }
                        assertEquals(expected, packed[i, j])
                    }
                }
            }
            val transposed = DoubleArray(9)
            ScalarLayoutKernels.transpose(source, MatrixWindow(transposed, 3, 3))
            for (j in 0 until 3) for (i in 0 until 3) assertEquals(source[i, j], transposed[j + i * 3])
        }
    }

    @Test
    fun `aliased packing unpacking and transpose stage original values`() {
        val data = DoubleArray(30) { it.toDouble() }
        val expected = data.copyOfRange(0, 6)
        val packed = ScalarLayoutKernels.packInto(
            MatrixWindow(data, 3, 2),
            data,
            PackedMatrixLayout(PackedRole.Right, 3, 2, 2),
        )
        ScalarLayoutKernels.unpack(packed, MatrixWindow(data, 3, 2, offset = 1))
        assertContentEquals(expected, data.copyOfRange(1, 7))
        ScalarLayoutKernels.transpose(MatrixWindow(data, 3, 2, offset = 1), MatrixWindow(data, 2, 3, offset = 1))
        for (j in 0 until 2) for (i in 0 until 3) assertEquals(expected[i + j * 3], data[1 + j + i * 2])
    }

    @Test
    fun `baked transforms are explicit and implicit triangular zeros remain positive`() {
        val source = MatrixWindow(
            doubleArrayOf(Double.NaN, -0.0, Double.NaN, Double.NaN),
            2,
            2,
            structure = MatrixStructure.UnitLower,
        )
        val packed = ScalarLayoutKernels.pack(
            source.transpose(),
            PackedMatrixLayout(PackedRole.Right, 2, 2, 4),
            Double.POSITIVE_INFINITY,
        )
        assertEquals(Double.POSITIVE_INFINITY, packed[0, 0])
        assertEquals(0L, packed[1, 0].toRawBits())
        assertEquals(Double.NaN, packed[0, 1])
        assertEquals(true, packed.bakedTranspose)
        assertEquals(Double.POSITIVE_INFINITY, packed.bakedScale)
        assertEquals(PackedOwnership.Owned, packed.ownership)
    }

    @Test
    fun `invalid packing shape leaves backing storage untouched`() {
        val data = DoubleArray(10) { -1.0 }
        assertFailsWith<IllegalArgumentException> {
            ScalarLayoutKernels.packInto(
                MatrixWindow(DoubleArray(4), 2, 2),
                data,
                PackedMatrixLayout(PackedRole.Left, 3, 2, 2),
            )
        }
        assertContentEquals(DoubleArray(10) { -1.0 }, data)
    }
}
