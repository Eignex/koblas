package com.eignex.koblas.internal.host

import java.lang.foreign.Arena

internal actual fun <T> withNativeBlocks(body: (NativeBlockAllocator) -> T): T =
    Arena.ofConfined().use { body(ArenaBlocks(it)) }
