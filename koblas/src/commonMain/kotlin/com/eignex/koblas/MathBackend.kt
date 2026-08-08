package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels

/**
 * Short human-readable identifier for the vector kernels the current process resolved: `"scalar"` on any
 * non-JVM target (or a JVM started without `--add-modules=jdk.incubator.vector`), `"simd(4 lanes)"` on a
 * JVM with AVX2, `"simd(8 lanes)"` with AVX-512, and a `"+openblas"` suffix when a host backend is
 * registered for long runs. Print at startup to verify your runtime picked up what you expected.
 *
 * Common rather than per-target, because [PlatformVectorKernels] names itself and this is just
 * `koblas.vectorKernels.name`. One expression cannot drift the way an `actual` per target can, where a JVM
 * one computed eagerly would keep reporting the compiled kernels after a host backend registered.
 */
public val mathBackend: String get() = koblas.vectorKernels.name
