package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Whether the incubating Vector API resolved without initializing its implementation classes. */
internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

internal val cKernelsAvailable: Boolean = !simdAvailable && JvmCKernelBindings.isAvailable

internal val cDenseKernelFamilies: DenseKernelFamilies =
    DenseKernelFamilies(CKernels, CPanelKernels, CPackedKernels)

internal val simdDenseKernelFamilies: DenseKernelFamilies =
    DenseKernelFamilies(SimdKernels, SimdPanelKernels, SimdPackedKernels)

internal actual val platformDenseKernelFamilies: DenseKernelFamilies = when {
    simdAvailable -> simdDenseKernelFamilies
    cKernelsAvailable -> cDenseKernelFamilies
    else -> scalarDenseKernelFamilies
}

/** JVM vector family selected once with the complete platform composition. */
internal actual object PlatformVectorKernels : DenseVectorKernels by platformDenseKernelFamilies.vector
