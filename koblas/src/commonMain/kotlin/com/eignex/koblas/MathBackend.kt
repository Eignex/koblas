package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels

/**
 * Short identifier for the vector kernels the current process resolved, as named by
 * [PlatformVectorKernels]: `"c"`, `"simd(8 lanes)"`, or `"scalar"`.
 */
public val mathBackend: String get() = koblas.vectorKernels.name
