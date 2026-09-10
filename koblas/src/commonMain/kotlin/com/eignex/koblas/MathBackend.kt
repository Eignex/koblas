package com.eignex.koblas

/**
 * Short identifier for the vector kernels the current process resolved: `"c"`, `"simd(8 lanes)"`, or
 * `"scalar"`.
 */
public val mathBackend: String get() = koblas.vectorKernels.name
