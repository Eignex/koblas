package com.eignex.koblas.internal.kernels

/** Complete portable arithmetic; individual backends implement only the roles they supply. */
internal interface EngineKernels : VectorKernels, IndexedKernels, PanelKernels, PackedProductKernels, TriangularKernels
