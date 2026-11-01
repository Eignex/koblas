package com.eignex.koblas

import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.host.HostDenseCall
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.vendor.Blas

/** Backends declare positive vector capabilities; structured families and binding exports add their own. */
internal class BackendContributions(
    override val name: String,
    val vectors: DenseVectorKernels = ScalarVectorKernels,
    vectorImplementations: Set<EngineImplementation> = emptySet(),
    val indexedDot: IndexedSparseKernels? = null,
    val panels: DensePanelKernels? = null,
    val products: DenseProductKernels? = null,
    val triangular: DenseTriangularKernels? = null,
    val binding: Blas? = null,
    val vectorName: String = name,
    val dynamicNorm: Boolean = false,
    val contiguousVectors: Boolean = false,
    val minimumVectorLength: Int = 0,
    val minimumIndexedLength: Int = 0,
) : EngineBackend {
    override val implementations: Set<EngineImplementation> = buildSet {
        addAll(vectorImplementations)
        if (indexedDot != null) add(EngineImplementation.IndexedDot)
        if (panels != null) add(EngineImplementation.Panels)
        if (products != null) add(EngineImplementation.Products)
        if (triangular != null) add(EngineImplementation.Triangular)
        if (binding != null) {
            for (operation in EngineImplementation.entries) {
                if (HostDenseCall.entryPointFor(operation) in binding.directlyImplemented) {
                    add(operation)
                }
            }
        }
        if (name == "portable") addAll(EngineImplementation.entries)
    }
}

/** Shared schedules and portable interface defaults provide the intrinsic availability floor. */
internal val portableBackend: BackendContributions = BackendContributions(
    "portable",
    panels = PortablePanelKernels,
    products = PortableProductKernels,
    triangular = PortableTriangularKernels,
    vectorName = "scalar",
)
