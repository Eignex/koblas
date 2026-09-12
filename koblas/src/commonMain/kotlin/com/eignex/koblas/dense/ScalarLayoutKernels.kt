package com.eignex.koblas.dense

/**
 * Independently callable scalar packing and layout reference. Copies preserve bits when scale is one;
 * structural zeroes and all declared physical padding are positive zero. Aliases use one logical snapshot
 * before any write. Nonalias calls into caller-owned storage allocate no scratch.
 */
public object ScalarLayoutKernels {
    /** Allocates and packs a retained operand whose layout does not depend on the engine or current thread. */
    public fun pack(source: MatrixWindow, layout: PackedMatrixLayout, scale: Double = 1.0): PackedMatrix {
        requireShape(source, layout)
        val destination = PackedMatrix.owned(layout, scale, source.transposed)
        packValues(source, destination)
        return destination
    }

    /**
     * Packs into caller-owned storage, clearing the declared padding, and returns its retained descriptor.
     * All validation and any alias snapshot finish before mutation. Scaling zero performs arithmetic on stored
     * entries; this layout operation has no BLAS zero-alpha shortcut. Implicit triangular zeroes remain zero.
     */
    public fun packInto(
        source: MatrixWindow,
        data: DoubleArray,
        layout: PackedMatrixLayout,
        offset: Int = 0,
        scale: Double = 1.0,
    ): PackedMatrix {
        requireShape(source, layout)
        val destination = PackedMatrix.wrap(data, layout, offset, scale, source.transposed)
        packValues(source, destination)
        return destination
    }

    /** Writes valid logical entries; physical padding is not read. Selected-output rules apply after transpose. */
    public fun unpack(
        source: PackedMatrix,
        destination: MatrixWindow,
        transpose: Boolean = false,
        output: BlockOutput = BlockOutput.Full,
    ) {
        copyValues(source, destination, transpose, output)
    }

    /** Transposes a general, symmetric, or triangular source into a general output window. */
    public fun transpose(source: MatrixWindow, destination: MatrixWindow, output: BlockOutput = BlockOutput.Full) {
        copyValues(source, destination, true, output)
    }

    private fun packValues(source: MatrixWindow, destination: PackedMatrix) {
        val stable = if (source.buffer === destination.buffer) snapshotOperand(source) else source
        val layout = destination.layout
        destination.buffer.fill(0.0, destination.offset, destination.offset + layout.storageSize)
        for (j in 0 until source.columns) {
            for (i in 0 until source.rows) {
                val value = stable.value(i, j)
                destination.buffer[destination.offset + layout.index(i, j)] = when {
                    source.implicitZero(i, j) -> 0.0
                    destination.bakedScale == 1.0 -> value
                    else -> destination.bakedScale * value
                }
            }
        }
    }

    private fun copyValues(source: MatrixOperand, destination: MatrixWindow, transpose: Boolean, output: BlockOutput) {
        require(destination.structure == MatrixStructure.General) { "output window must be general" }
        require(
            destination.rows == (if (transpose) source.columns else source.rows) &&
                destination.columns == (if (transpose) source.rows else source.columns),
        ) { "layout copy shape mismatch" }
        val stable = if (source.buffer === destination.buffer) snapshotOperand(source) else source
        for (j in 0 until destination.columns) {
            for (i in 0 until destination.rows) {
                if (output.contains(i, j)) {
                    destination.data[
                        destination.index(
                            i,
                            j,
                        ),
                    ] = if (transpose) stable.value(j, i) else stable.value(i, j)
                }
            }
        }
    }

    private fun requireShape(source: MatrixWindow, layout: PackedMatrixLayout) {
        require(source.rows == layout.rows && source.columns == layout.columns) { "packing shape mismatch" }
    }
}

internal fun snapshotOperand(source: MatrixOperand): MatrixWindow {
    val data = DoubleArray(checkedStorageSize(source.rows.toLong() * source.columns))
    for (j in 0 until source.columns) {
        for (i in 0 until source.rows) data[i + j * source.rows] = source.value(i, j)
    }
    return MatrixWindow(data, source.rows, source.columns)
}
