package com.eignex.koblas.bench

import com.eignex.koblas.*

/** Reject engines that do not supply the requested vendor entry point. */
internal fun exactArmRejection(
    backend: KoblasEngine,
    operation: EngineImplementation,
): String? {
    if (backend.vendor == null) return "Backend has no vendor"
    if (operation !in backend.implementations) return "Backend does not supply $operation"
    return null
}
