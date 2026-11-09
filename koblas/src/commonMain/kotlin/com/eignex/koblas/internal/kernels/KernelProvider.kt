package com.eignex.koblas.internal.kernels

import com.eignex.koblas.KoblasEngine

/** An engine that supplies numerical callbacks as well as the public operation contract. */
internal interface KernelProvider :
    KoblasEngine,
    EngineKernels

internal fun KoblasEngine.unsupported(): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")

/** Only built-in providers expose primitive arithmetic; custom public engines may override typed calls. */
internal val KoblasEngine.kernels: KernelProvider get() = this as? KernelProvider ?: unsupported()
