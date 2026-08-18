package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra

/** The JVM discovers OpenBLAS through the ServiceLoader, so auto needs no explicit install. */
internal actual fun nativeBackend(): F64LinearAlgebra? = null

/** The JVM's level-1 kernels are its SIMD ones, so there is nothing to swap in. */
internal actual fun useHostLevel1(enabled: Boolean): Boolean = false
