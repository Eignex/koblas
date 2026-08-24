package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra

internal actual fun nativeBackend(): F64LinearAlgebra? = null

internal actual fun useHostLevel1(enabled: Boolean): Boolean = false
