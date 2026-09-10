package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals

class PackedTileOutputTest {
    @Test
    fun `masked accumulation changes only the selected triangle`() {
        val kernels = fixedTile(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val destination = doubleArrayOf(10.0, 20.0, 30.0, 40.0)
        val scratch = DoubleArray(4)

        accumulatePackedProductTile(
            kernels,
            depth = 1,
            packedA = doubleArrayOf(1.0, 1.0),
            aOffset = 0,
            packedB = doubleArrayOf(1.0, 1.0),
            bOffset = 0,
            destination = destination,
            destinationOffset = 0,
            leadingDimension = 2,
            validRows = 2,
            validColumns = 2,
            destinationRow = 0,
            destinationColumn = 0,
            triangle = true,
            tile = scratch,
        )

        assertEquals(listOf(11.0, 22.0, 30.0, 44.0), destination.toList())
    }

    @Test
    fun `masked accumulation skips an outside tile without reads`() {
        var calls = 0
        val kernels = object : PackedKernels by fixedTile(DoubleArray(4)) {
            override fun gemmTile(
                depth: Int,
                packedA: DoubleArray,
                aOff: Int,
                packedB: DoubleArray,
                bOff: Int,
                c: DoubleArray,
                cOff: Int,
                ldc: Int,
            ) {
                calls++
            }
        }
        val destination = DoubleArray(4) { Double.NaN }

        accumulatePackedProductTile(
            kernels, 1, DoubleArray(0), 0, DoubleArray(0), 0, destination, 0, 2,
            2, 2, destinationRow = 0, destinationColumn = 2, triangle = true, DoubleArray(4),
        )

        assertEquals(0, calls)
        destination.forEach { assertEquals(Double.NaN, it) }
    }

    @Test
    fun `edge overwrite replaces valid entries and preserves outside entries`() {
        val kernels = fixedTile(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val destination = DoubleArray(9) { 99.0 }

        writePackedProductTile(
            kernels, 1, doubleArrayOf(1.0, 1.0), 0, doubleArrayOf(1.0, 1.0), 0,
            destination, destinationOffset = 1, leadingDimension = 3,
            validRows = 1, validColumns = 2, tile = DoubleArray(4),
        )

        assertEquals(1.0, destination[1])
        assertEquals(3.0, destination[4])
        for (index in destination.indices) {
            if (index != 1 && index != 4) assertEquals(99.0, destination[index], "index=$index")
        }
    }

    private fun fixedTile(values: DoubleArray): PackedKernels = object : PackedKernels by PortablePackedKernels {
        override val gemmTileRows: Int = 2
        override val gemmTileCols: Int = 2

        override fun gemmTile(
            depth: Int,
            packedA: DoubleArray,
            aOff: Int,
            packedB: DoubleArray,
            bOff: Int,
            c: DoubleArray,
            cOff: Int,
            ldc: Int,
        ) {
            var column = 0
            while (column < gemmTileCols) {
                var row = 0
                while (row < gemmTileRows) {
                    c[cOff + row + column * ldc] += values[row + column * gemmTileRows]
                    row++
                }
                column++
            }
        }
    }
}
