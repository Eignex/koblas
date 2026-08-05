package com.eignex.koblas.dense

import com.eignex.koblas.registerBackend

/**
 * Registers whatever backends this platform provides.
 *
 * The single discovery hook. The JVM's actual scans the classpath with `ServiceLoader` and registers its
 * host-OpenBLAS halves; the native targets register eagerly before `main` instead (see the cblas
 * AutoInstall), which is earlier than this would run, so theirs is a no-op; the web targets have nothing
 * to reach. Whatever it finds goes through [registerBackend] like any other backend, so there is one
 * ranking and one fallback rather than a parallel path.
 */
internal expect fun registerPlatformBackends()
