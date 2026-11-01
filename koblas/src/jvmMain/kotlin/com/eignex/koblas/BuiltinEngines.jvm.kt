package com.eignex.koblas

import com.eignex.koblas.vendor.Blas

/** Owned JVM kernels, with eligible host calls when native access and a measured vendor are available. */
internal actual fun platformEngine(): KoblasEngine = JvmBackend.default

internal actual fun simdBackend(): BackendContributions? = JvmBackend.backend

internal actual fun vendorBackend(binding: Blas): BackendContributions = BackendContributions(
    binding.vendor.vendorName.lowercase(),
    com.eignex.koblas.internal.host.BindingVectorKernels(binding),
    binding = binding,
    minimumVectorLength = 1,
    vectorName = "${binding.vendor.vendorName.lowercase()}-level1",
)
