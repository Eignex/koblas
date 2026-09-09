package com.eignex.koblas.dense

/** One immutable dense implementation assembled from the three numerical responsibilities. */
internal class DenseKernelFamilies(
    val vector: DenseVectorKernels,
    val panel: DensePanelKernels,
    val packed: PackedKernels,
)

/** The scalar family composition shared by reference engines and platform fallbacks. */
internal val scalarDenseKernelFamilies: DenseKernelFamilies = DenseKernelFamilies(
    vector = ScalarKernels,
    panel = ScalarPanelKernels,
    packed = PortablePackedKernels,
)

/** The immutable family composition selected once for the current platform. */
internal expect val platformDenseKernelFamilies: DenseKernelFamilies
