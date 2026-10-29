@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import kotlinx.cinterop.CPointer
import platform.posix.FILE

internal expect fun createDirectory(path: String)
internal expect fun openTextFile(path: String, mode: String): CPointer<FILE>?
