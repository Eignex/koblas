package com.eignex.koblas

import com.eignex.koblas.internal.host.BindingVectorKernels
import com.eignex.koblas.internal.host.NativeVendorBlas
import com.eignex.koblas.vendor.Blas

internal actual fun simdBackend(): BackendContributions? = null

internal actual fun vendorBackend(binding: Blas): BackendContributions {
    val vectors = if (binding is NativeVendorBlas) binding else BindingVectorKernels(binding)
    return BackendContributions(
        binding.vendor.vendorName.lowercase(),
        vectors,
        binding = binding,
        vectorName = vectors.name,
        minimumVectorLength = 1,
    )
}

/** Select an applicable measured profile once, with portable coverage when none matches. */
internal actual fun platformEngine(): KoblasEngine = standardEngine("native")
