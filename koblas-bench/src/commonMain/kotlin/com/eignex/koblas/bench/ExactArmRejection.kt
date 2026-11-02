package com.eignex.koblas.bench

import com.eignex.koblas.*

/** Reject absent exports and calls whose own quick return reaches no foreign arithmetic. */
internal fun exactArmRejection(
    backend: KoblasEngine,
    operation: EngineImplementation,
    matrices: List<DenseMatrixRegion> = emptyList(),
    vectors: List<DenseVector> = emptyList(),
    alpha: Double = 1.0,
): String? {
    if (backend.vendor == null) return "Backend has no vendor"
    if (operation !in backend.implementations) return "Backend does not export ${operation.entryPoint}"
    if (vectors.any { it.size == 0 } || matrices.any { it.rows == 0 || it.cols == 0 }) {
        return "Call does not reach the requested vendor entry point"
    }
    if (operation.level >= 2 && alpha == 0.0) return "Zero product multiplier uses shared scaling instead of the requested entry point"
    return null
}
