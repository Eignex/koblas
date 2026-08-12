package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels

/**
 * Short identifier for the vector kernels the current process resolved, as named by
 * [PlatformVectorKernels]: `"scalar"`, `"simd(8 lanes)"`, or a `"+openblas"` suffix for a host backend.
 */
public val mathBackend: String get() = koblas.vectorKernels.name
