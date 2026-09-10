package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinKernels

internal val ReferenceBlas: Blas = BuiltinKernels.scalar

internal fun testDenseKernelFamilies(
    vector: DenseVectorKernels = ScalarKernels,
    panel: DensePanelKernels = ScalarPanelKernels,
    packed: PackedKernels = PortablePackedKernels,
): DenseKernelFamilies = DenseKernelFamilies(vector, panel, packed)
