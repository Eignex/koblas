package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.PackedKernels
import com.eignex.koblas.internal.kernels.JvmCKernelBindings

// This development module calls the existing bindings to exclude the production depth crossover.
// Keep this access local to the baseline harness; the planned exact kernel interface will replace it.
internal actual fun benchmarkPackedKernels(engine: KoblasEngine): PackedKernels =
    if (engine === BuiltinEngines.c) ExactCPackedKernels else engine.packedKernels

private object ExactCPackedKernels : PackedKernels by requireNotNull(BuiltinEngines.c).packedKernels {
    override fun gemmTile(depth: Int, packedA: DoubleArray, aOff: Int, packedB: DoubleArray, bOff: Int,
        c: DoubleArray, cOff: Int, ldc: Int) {
        JvmCKernelBindings.denseGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
    }

    override fun gemmTrsmTile(depth: Int, validRows: Int, order: Int, packedA: DoubleArray, aOff: Int,
        packedB: DoubleArray, bOff: Int, packedTriangle: DoubleArray, triangleOff: Int, lower: Boolean,
        unitDiag: Boolean, x: DoubleArray, xOff: Int) {
        JvmCKernelBindings.denseGemmTrsmTile(depth, validRows, order, packedA, aOff, packedB, bOff,
            packedTriangle, triangleOff, lower, unitDiag, x, xOff)
    }
}
