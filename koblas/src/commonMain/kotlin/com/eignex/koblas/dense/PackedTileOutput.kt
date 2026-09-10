package com.eignex.koblas.dense

/**
 * Trusted output leaves for a packed product tile and its non-empty logical destination edge. Callers provide
 * valid panel, tile and destination windows; the leaves own only full-tile dispatch and edge writeback arithmetic.
 */

@Suppress("LongParameterList")
internal fun accumulatePackedProductTile(
    kernels: PackedKernels,
    depth: Int,
    packedA: DoubleArray,
    aOffset: Int,
    packedB: DoubleArray,
    bOffset: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    leadingDimension: Int,
    validRows: Int,
    validColumns: Int,
    destinationRow: Int,
    destinationColumn: Int,
    triangle: Boolean?,
    tile: DoubleArray,
) {
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    val lastRow = destinationRow + validRows - 1
    val lastColumn = destinationColumn + validColumns - 1
    val outside = triangle != null && if (triangle) {
        lastRow < destinationColumn
    } else {
        destinationRow > lastColumn
    }
    if (outside) return

    val inside = triangle == null || if (triangle) {
        destinationRow >= lastColumn
    } else {
        lastRow <= destinationColumn
    }
    if (inside && validRows == tileRows && validColumns == tileColumns) {
        kernels.gemmTile(
            depth,
            packedA,
            aOffset,
            packedB,
            bOffset,
            destination,
            destinationOffset,
            leadingDimension,
        )
        return
    }

    tile.fill(0.0, 0, tileRows * tileColumns)
    kernels.gemmTile(depth, packedA, aOffset, packedB, bOffset, tile, 0, tileRows)
    var column = 0
    while (column < validColumns) {
        val source = column * tileRows
        val target = destinationOffset + column * leadingDimension
        var row = 0
        while (row < validRows) {
            val selected = triangle == null || if (triangle) {
                destinationRow + row >= destinationColumn + column
            } else {
                destinationRow + row <= destinationColumn + column
            }
            if (selected) destination[target + row] += tile[source + row]
            row++
        }
        column++
    }
}

@Suppress("LongParameterList")
internal fun writePackedProductTile(
    kernels: PackedKernels,
    depth: Int,
    packedA: DoubleArray,
    aOffset: Int,
    packedB: DoubleArray,
    bOffset: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    leadingDimension: Int,
    validRows: Int,
    validColumns: Int,
    tile: DoubleArray,
) {
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    if (validRows == tileRows && validColumns == tileColumns) {
        kernels.gemmTile(
            depth,
            packedA,
            aOffset,
            packedB,
            bOffset,
            destination,
            destinationOffset,
            leadingDimension,
        )
        return
    }

    tile.fill(0.0, 0, tileRows * tileColumns)
    kernels.gemmTile(depth, packedA, aOffset, packedB, bOffset, tile, 0, tileRows)
    var column = 0
    while (column < validColumns) {
        tile.copyInto(
            destination,
            destinationOffset + column * leadingDimension,
            column * tileRows,
            column * tileRows + validRows,
        )
        column++
    }
}
