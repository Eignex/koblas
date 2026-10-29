@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import kotlinx.cinterop.CPointer
import platform.posix.FILE
import platform.posix.fopen
import platform.posix.mkdir

internal actual fun createDirectory(path: String) {
    mkdir(path, 0x1ffu)
}

internal actual fun openTextFile(path: String, mode: String): CPointer<FILE>? = fopen(path, mode)
