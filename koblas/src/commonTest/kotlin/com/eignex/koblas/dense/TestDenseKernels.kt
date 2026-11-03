package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine

/** Runs each available panel implementation once, including the mandatory portable implementation. */
internal fun withPanelKernels(body: (KoblasEngine) -> Unit) {
    listOfNotNull(PortableEngine, koblas, BuiltinEngines.simd)
        .map { it.panelProvider }.distinct().forEach(body)
}

/** Runs each available product implementation once, including the mandatory portable implementation. */
internal fun withProductKernels(body: (KoblasEngine) -> Unit) {
    listOfNotNull(PortableEngine, koblas, BuiltinEngines.simd)
        .map { it.productProvider }.distinct().forEach(body)
}
