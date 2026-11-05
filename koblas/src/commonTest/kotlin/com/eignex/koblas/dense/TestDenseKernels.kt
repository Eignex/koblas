package com.eignex.koblas.dense

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.koblas

/** Runs each available panel implementation once, including the mandatory portable implementation. */
internal fun withPanelKernels(body: (EngineKernels) -> Unit) {
    listOfNotNull(ScalarEngine, koblas, KoblasEngine.simd)
        .map { it.panelProvider }.distinct().forEach(body)
}

/** Runs each available product implementation once, including the mandatory portable implementation. */
internal fun withProductKernels(body: (EngineKernels) -> Unit) = productKernels().forEach(body)

/** Each available product implementation once, which is what decides the block geometry a schedule uses. */
internal fun productKernels(): List<EngineKernels> =
    listOfNotNull(ScalarEngine, koblas, KoblasEngine.simd).map { it.productProvider }.distinct()
