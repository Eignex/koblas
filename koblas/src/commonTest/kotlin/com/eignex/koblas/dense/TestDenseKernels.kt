package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.PortableProductKernels
import com.eignex.koblas.koblas

/** Runs each available panel implementation once, including the mandatory portable implementation. */
internal fun withPanelKernels(body: (KoblasEngine) -> Unit) {
    listOfNotNull(PortablePanelKernels, koblas, BuiltinEngines.simd)
        .distinct().forEach(body)
}

/** Runs each available product implementation once, including the mandatory portable implementation. */
internal fun withProductKernels(body: (KoblasEngine) -> Unit) {
    listOfNotNull(PortableProductKernels, koblas, BuiltinEngines.simd)
        .distinct().forEach(body)
}
