package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels

/** Exact indexed composition shares the engine's compiled routing decisions. */
internal val SimdIndexedSparseKernels: IndexedSparseKernels get() =
    BuiltinEngines.simd?.indexedSparseKernels ?: ScalarIndexedSparseKernels
