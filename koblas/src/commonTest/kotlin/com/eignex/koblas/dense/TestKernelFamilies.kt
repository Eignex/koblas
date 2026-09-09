package com.eignex.koblas.dense

internal fun testDenseKernelFamilies(
    vector: DenseVectorKernels = ScalarKernels,
    panel: DensePanelKernels = ScalarPanelKernels,
    packed: PackedKernels = PortablePackedKernels,
): DenseKernelFamilies = DenseKernelFamilies(vector, panel, packed)
