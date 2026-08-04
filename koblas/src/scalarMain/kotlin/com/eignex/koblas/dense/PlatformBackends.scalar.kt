package com.eignex.koblas.dense

/**
 * Nothing to do off the JVM.
 *
 * The native targets register their host-BLAS backend eagerly before `main` (see the cblas AutoInstall),
 * which is earlier than this would run, and the JS and Wasm targets have no native library to reach.
 */
internal actual fun registerPlatformBackends() = Unit
