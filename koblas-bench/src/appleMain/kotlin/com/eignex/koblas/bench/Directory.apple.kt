@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import platform.posix.mkdir

internal actual fun createDirectory(path: String) {
    mkdir(path, 0x1ffu)
}
