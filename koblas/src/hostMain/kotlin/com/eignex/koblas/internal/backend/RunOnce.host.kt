package com.eignex.koblas.internal.backend

/**
 * No lock. The standard library offers these targets nothing a thread may re-enter, and a lock that cannot
 * be re-entered would hang the case [RunOnce] exists to serve, where the pass asks for itself from inside.
 *
 * What is left unguarded is narrow: [registerPlatformBackends] here opens koblas's own bindings and runs no
 * code a caller supplied, so two threads reaching it together settle on the same backends whichever order
 * they take. The JVM, where a service-loaded provider runs inside the pass, has a monitor and uses it.
 */
internal actual fun <T> withRunOnceLock(block: () -> T): T = block()
