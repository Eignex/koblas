package com.eignex.koblas

// Linux and macOS, the targets that can reach a host BLAS. Runs of at least HOST_LEVEL1_MIN_LENGTH go to
// installed kernels when there are any; everything else takes the scalar loop. The length is compared
// first, against a compile-time constant, so a short run never reads the installed-kernels global.
//
// Measured on linuxX64 in a release binary, host OpenBLAS against the scalar loops (ns/op):
//
//     len        2      16      32      64     256    1024    4096
//     dot     36.7    44.4    38.0    39.6    44.4   108.0   443.2
//     scalar  15.8    23.0    35.6    59.1   193.8   719.3  2888.5
//     axpy    34.3    35.6    35.0    36.8    44.2    82.8   471.1
//     scalar  10.4    22.5    32.5    57.7   227.0   886.8  3326.9
//
// A call costs roughly 35ns flat, so it loses below 32, breaks even there for axpy and at 64 for dot,
// and from then on wins by 1.4x to 10x. The threshold is the higher crossover of the two, so neither
// operation regresses.

internal actual fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    if (len >= HOST_LEVEL1_MIN_LENGTH) {
        installedLevel1Kernels()?.let { return it.dot(a, aOff, b, bOff, len) }
    }
    return scalarDot(a, aOff, b, bOff, len)
}

internal actual fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    if (len >= HOST_LEVEL1_MIN_LENGTH) {
        installedLevel1Kernels()?.let {
            it.axpy(y, yOff, alpha, x, xOff, len)
            return
        }
    }
    scalarAxpy(y, yOff, alpha, x, xOff, len)
}

internal actual fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    if (len >= HOST_LEVEL1_MIN_LENGTH) {
        installedLevel1Kernels()?.let {
            it.scale(v, vOff, alpha, len)
            return
        }
    }
    scalarScale(v, vOff, alpha, len)
}
