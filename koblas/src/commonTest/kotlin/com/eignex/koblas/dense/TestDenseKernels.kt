package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.koblas

/** Runs each available panel implementation once, including the mandatory portable implementation. */
@OptIn(KoblasEngineApi::class)
internal fun withPanelKernels(body: (DensePanelKernels) -> Unit) {
    listOfNotNull(PortablePanelKernels, koblas.panelKernels, BuiltinEngines.simd?.panelKernels)
        .distinct().forEach(body)
}

/** Runs each available product implementation once, including the mandatory portable implementation. */
@OptIn(KoblasEngineApi::class)
internal fun withProductKernels(body: (DenseProductKernels) -> Unit) {
    listOfNotNull(PortableProductKernels, koblas.productKernels, BuiltinEngines.simd?.productKernels)
        .distinct().forEach(body)
}
