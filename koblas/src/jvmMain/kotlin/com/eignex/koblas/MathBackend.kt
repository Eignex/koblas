package com.eignex.koblas

import com.eignex.koblas.dense.Simd
import com.eignex.koblas.dense.simdAvailable

/** Identifies the runtime math backend powering the SIMD-like primitives. */
public actual val mathBackend: String = if (simdAvailable) "simd(${Simd.lanes()} lanes)" else "scalar"
