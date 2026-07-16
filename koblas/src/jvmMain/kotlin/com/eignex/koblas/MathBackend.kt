package com.eignex.koblas

/** Identifies the runtime math backend powering the SIMD-like primitives. */
public actual val mathBackend: String = if (simdAvailable) "simd(${Simd.lanes()} lanes)" else "scalar"
