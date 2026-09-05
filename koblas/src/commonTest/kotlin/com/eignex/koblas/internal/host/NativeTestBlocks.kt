package com.eignex.koblas.internal.host

/**
 * Runs [body] against native memory that goes away when it returns, so struct marshalling can be checked on
 * either platform without a library to call.
 */
internal expect fun <T> withNativeBlocks(body: (NativeBlockAllocator) -> T): T
