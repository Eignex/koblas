package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.internal.dense.DensePanelKernels
import com.eignex.koblas.internal.dense.DenseProductKernels
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.PortableProductKernels
import com.eignex.koblas.koblas

/** Runs each available panel implementation once, including the mandatory portable implementation. */
internal fun withPanelKernels(body: (DensePanelKernels) -> Unit) {
    listOfNotNull(PortablePanelKernels, koblas.panelKernels, BuiltinEngines.simd?.panelKernels)
        .distinct().forEach(body)
}

/** Runs each available product implementation once, including the mandatory portable implementation. */
internal fun withProductKernels(body: (DenseProductKernels) -> Unit) {
    listOfNotNull(PortableProductKernels, koblas.productKernels, BuiltinEngines.simd?.productKernels)
        .distinct().forEach(body)
}
