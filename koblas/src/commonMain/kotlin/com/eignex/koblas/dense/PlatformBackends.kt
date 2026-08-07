package com.eignex.koblas.dense

import com.eignex.koblas.registerBackend

/**
 * Registers whatever backends this platform provides.
 *
 * The single discovery hook, called once on the first [com.eignex.koblas.koblas] read on every platform.
 * The JVM's actual registers its host bindings and then scans the classpath with `ServiceLoader`; Linux and
 * macOS register theirs; the remaining native targets and the web ones have no host library to reach and do
 * nothing. Whatever it finds goes through [registerBackend] like any other backend, so there is one ranking
 * and one fallback rather than a parallel path.
 */
internal expect fun registerPlatformBackends()
