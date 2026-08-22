package com.eignex.koblas.internal.backend

import com.eignex.koblas.registerBackend

/**
 * Registers whatever backends this platform provides, once, on the first [com.eignex.koblas.koblas] read.
 * Everything it finds goes through [registerBackend] like any other backend.
 */
internal expect fun registerPlatformBackends()
