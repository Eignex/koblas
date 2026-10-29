@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.host

import com.eignex.koblas.vendor.*
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.invoke
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toKString
import platform.posix.RTLD_NOW
import platform.posix.dlclose
import platform.posix.dlopen
import platform.posix.dlsym
import platform.posix.setenv

private typealias DlAddrFn = CFunction<(COpaquePointer?, CPointer<ByteVar>?) -> Int>

/** POSIX loader shared by Linux and Apple targets. */
internal actual class NativeLibrary private constructor(private val handle: COpaquePointer) {
    actual fun symbol(name: String): COpaquePointer? = dlsym(handle, name)

    actual fun symbolOwner(name: String): String? = memScoped {
        val address = symbol(name) ?: return@memScoped null
        // dladdr is a GNU extension not surfaced by the POSIX platform bindings.
        val dladdr = symbol("dladdr")?.reinterpret<DlAddrFn>() ?: return@memScoped null
        val info = allocArray<ByteVar>(DL_INFO_BYTES)
        if (dladdr(address, info) == 0) return@memScoped null
        val path = info.reinterpret<CPointerVar<ByteVar>>()[0] ?: return@memScoped null
        path.toKString().ifEmpty { null }
    }

    actual fun close() {
        dlclose(handle)
    }

    actual companion object {
        actual fun open(path: String): NativeLibrary? = dlopen(path, RTLD_NOW)?.let { NativeLibrary(it) }

        /** Dl_info is four pointers, of which only dli_fname is read. */
        private const val DL_INFO_BYTES = 32
    }
}

internal actual fun setNativeEnvironmentVariable(name: String, value: String) {
    setenv(name, value, 1)
}
