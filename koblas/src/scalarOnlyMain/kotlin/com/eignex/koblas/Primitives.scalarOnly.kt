package com.eignex.koblas

// The targets with no host BLAS to reach for: JS, Wasm, iOS and Windows. They call the scalar loops
// directly, with no dispatch of any kind, so nothing here can cost them anything at runtime.

internal actual fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
    scalarDot(a, aOff, b, bOff, len)

internal actual fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
    scalarAxpy(y, yOff, alpha, x, xOff, len)

internal actual fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)
