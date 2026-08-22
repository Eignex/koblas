package com.eignex.koblas.internal.backend

/** No environment to read from a browser or a Wasm sandbox, and no native backend to tune either. */
internal actual fun dispatchOverride(element: ElementType, level: DispatchLevel): Int? = null
