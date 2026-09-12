package com.eignex.koblas.dense

/** Describes this composition, including stages whose implementation is known only from operand data. */
internal fun describeSimdComponent(operation: DenseOperation, length: Int): String {
    val vector = SimdVectorKernels.name
    if (operation.packed) {
        val geometry = "packed ${SimdPackedKernels.gemmTileRows}x${SimdPackedKernels.gemmTileCols}"
        return when (operation) {
            DenseOperation.TrsmTile -> "portable-solve; $geometry"

            DenseOperation.GemmTrsmTile -> if (length == 0) {
                "portable-solve; $geometry"
            } else {
                "$vector update + portable-solve; $geometry"
            }

            else -> "$vector product; $geometry"
        }
    }
    if (length < SimdOps.lanes() ||
        (operation == DenseOperation.Iamax && length < DenseTuning.simdIamaxCrossover)
    ) {
        return "scalar runtime (short input)"
    }
    val fallback = when (operation) {
        DenseOperation.Nrm2 -> "; scalar robust norm fallback depends on values"
        DenseOperation.DotAxpy -> "; scalar ordered dot fallback depends on values"
        DenseOperation.Rotm -> "; scalar fallback for nonunit strides"
        else -> ""
    }
    return "$vector with scalar tails$fallback"
}
