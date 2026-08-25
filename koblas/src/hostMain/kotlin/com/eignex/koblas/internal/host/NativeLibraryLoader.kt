@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.host

import kotlinx.cinterop.COpaquePointer
import platform.posix.*

/** Opens the first library in [names], optionally requiring that it exports [requiredSymbol]. */
internal fun openNativeLibrary(names: List<String>, requiredSymbol: String? = null): COpaquePointer? {
    for (name in names) {
        val opened = dlopen(name, RTLD_NOW) ?: continue
        if (requiredSymbol == null || dlsym(opened, requiredSymbol) != null) return opened
    }
    return null
}
