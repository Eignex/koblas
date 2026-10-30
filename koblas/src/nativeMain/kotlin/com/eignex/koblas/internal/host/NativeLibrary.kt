@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.host

import com.eignex.koblas.vendor.*
import kotlinx.cinterop.COpaquePointer

/** A platform loader handle whose symbols belong to the requested library. */
internal expect class NativeLibrary {
    fun symbol(name: String): COpaquePointer?
    fun symbolOwner(name: String): String?
    fun close()

    companion object {
        fun open(path: String): NativeLibrary?
    }
}

/** Set a process environment variable before the library performs arithmetic. */
internal expect fun setNativeEnvironmentVariable(name: String, value: String)
