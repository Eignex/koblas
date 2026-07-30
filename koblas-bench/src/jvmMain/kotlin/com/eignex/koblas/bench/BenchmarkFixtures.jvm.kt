package com.eignex.koblas.bench

import com.eignex.koblas.LinearAlgebra

/** The JVM discovers OpenBLAS through the ServiceLoader, so `auto` needs no explicit install. */
internal actual fun nativeBackend(): LinearAlgebra? = null
