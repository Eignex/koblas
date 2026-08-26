package com.eignex.koblas.internal.backend

/**
 * No lock, because there is nothing on these targets for one to guard: [registerPlatformBackends] discovers
 * nothing here, so a second caller has nothing half-assembled to see. JS and both Wasm flavours run one
 * thread besides.
 */
internal actual fun <T> withRunOnceLock(block: () -> T): T = block()
