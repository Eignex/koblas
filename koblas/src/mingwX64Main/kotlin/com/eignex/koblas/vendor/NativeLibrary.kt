@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.vendor

import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toKStringFromUtf16
import kotlinx.cinterop.value
import kotlinx.cinterop.wcstr
import platform.windows.FreeLibrary
import platform.windows.GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
import platform.windows.GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT
import platform.windows.GetModuleFileNameW
import platform.windows.GetModuleHandleExW
import platform.windows.GetProcAddress
import platform.windows.GetThreadErrorMode
import platform.windows.HMODULE
import platform.windows.HMODULEVar
import platform.windows.LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
import platform.windows.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
import platform.windows.LoadLibraryExW
import platform.windows.SEM_FAILCRITICALERRORS
import platform.windows.SetEnvironmentVariableW
import platform.windows.SetThreadErrorMode
import platform.windows.WCHARVar

/** Windows loader uses wide paths so installed libraries can live under Unicode directory names. */
internal actual class NativeLibrary private constructor(private val handle: HMODULE) {
    actual fun symbol(name: String): COpaquePointer? = GetProcAddress(handle, name)?.reinterpret()

    actual fun symbolOwner(name: String): String? = memScoped {
        val address = symbol(name) ?: return@memScoped null
        val owner = alloc<HMODULEVar>()
        val flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS or GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT
        if (GetModuleHandleExW(flags.toUInt(), address.reinterpret(), owner.ptr) == 0) return@memScoped null
        val path = allocArray<WCHARVar>(PATH_CHARACTERS)
        val length = GetModuleFileNameW(owner.value, path, PATH_CHARACTERS.toUInt())
        if (length == 0u || length >= PATH_CHARACTERS.toUInt()) return@memScoped null
        path.toKStringFromUtf16()
    }

    actual fun close() {
        FreeLibrary(handle)
    }

    actual companion object {
        actual fun open(path: String): NativeLibrary? = memScoped {
            val previous = GetThreadErrorMode()
            if (SetThreadErrorMode(previous or SEM_FAILCRITICALERRORS.toUInt(), null) == 0) return@memScoped null
            try {
                // A vendor's adjacent runtimes are searched without admitting the working directory.
                val flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS or
                    if (path.replace('\\', '/').isWindowsAbsolutePath()) LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR else 0
                LoadLibraryExW(path.replace('/', '\\').wcstr.ptr, null, flags.toUInt())?.let { NativeLibrary(it) }
            } finally {
                SetThreadErrorMode(previous, null)
            }
        }

        private const val PATH_CHARACTERS = 32768
    }
}

internal actual fun setNativeEnvironmentVariable(name: String, value: String) {
    memScoped { SetEnvironmentVariableW(name.wcstr.ptr, value.wcstr.ptr) }
}
