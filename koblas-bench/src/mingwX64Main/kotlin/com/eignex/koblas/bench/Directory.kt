@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import kotlinx.cinterop.CPointer
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.wcstr
import platform.windows.CreateDirectoryW
import platform.posix.FILE
import platform.posix._wfopen

internal actual fun createDirectory(path: String) {
    memScoped { CreateDirectoryW(path.wcstr.ptr, null) }
}

internal actual fun openTextFile(path: String, mode: String): CPointer<FILE>? = memScoped {
    _wfopen(path.wcstr.ptr, mode.wcstr.ptr)
}
