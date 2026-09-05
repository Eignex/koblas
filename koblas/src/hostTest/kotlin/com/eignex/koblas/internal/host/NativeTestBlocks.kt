@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.internal.host

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.memScoped

internal actual fun <T> withNativeBlocks(body: (NativeBlockAllocator) -> T): T = memScoped { body(ScopedBlocks(this)) }
