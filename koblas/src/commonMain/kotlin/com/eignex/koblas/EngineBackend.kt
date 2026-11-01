package com.eignex.koblas

import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.host.BindingVectorKernels
import com.eignex.koblas.internal.host.HostDenseCall
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.vendor.Blas

/** Available arithmetic and positive capabilities, independently of the policy selecting them. */
public class EngineBackend internal constructor(
    /** Stable identifier used in routing configurations. */
    public val name: String,
    internal val vectors: DenseVectorKernels = ScalarVectorKernels,
    implementations: Set<EngineImplementation> = emptySet(),
    internal val indexedDot: IndexedSparseKernels? = null,
    internal val panels: DensePanelKernels? = null,
    internal val products: DenseProductKernels? = null,
    internal val triangular: DenseTriangularKernels? = null,
    internal val binding: Blas? = null,
    internal val vectorName: String = name,
    internal val dynamicNorm: Boolean = false,
    internal val contiguousVectors: Boolean = false,
    internal val minimumVectorLength: Int = 0,
    internal val minimumIndexedLength: Int = 0,
) {
    /** Supplied capabilities, excluding unselected portable defaults. */
    public val implementations: Set<EngineImplementation> = buildSet {
        addAll(implementations)
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
    }
}

/** Arithmetic boundaries a routing configuration can select independently. */
public enum class EngineImplementation {
    /** Dense dot product. */
    Dot,

    /** Dense sum. */
    Sum,

    /** Dense rescaled norm. */
    Nrm2,

    /** Dense maximum magnitude index. */
    Iamax,

    /** Dense absolute sum. */
    Asum,

    /** Dense scaled addition. */
    Axpy,

    /** Dense scaling. */
    Scale,

    /** Dense exchange. */
    Swap,

    /** Dense rotation. */
    Rot,

    /** Indexed dot against a dense vector. */
    IndexedDot,

    /** Panel arithmetic within shared schedules. */
    Panels,

    /** Packed product arithmetic and its packing geometry. */
    Products,

    /** Diagonal substitution within triangular schedules. */
    Triangular,

    /** Whole matrix vector product. */
    Gemv,

    /** Whole transposed matrix vector product. */
    GemvTransposed,

    /** Whole symmetric matrix vector product. */
    Symv,

    /** Whole rank one update. */
    Ger,

    /** Whole symmetric rank one update. */
    Syr,

    /** Whole symmetric rank two update. */
    Syr2,

    /** Whole triangular matrix vector product. */
    Trmv,

    /** Whole transposed triangular matrix vector product. */
    TrmvTransposed,

    /** Whole triangular vector substitution. */
    Trsv,

    /** Whole transposed triangular vector substitution. */
    TrsvTransposed,

    /** Whole matrix product. */
    Gemm,

    /** Whole selected triangle product. */
    Gemmt,

    /** Whole symmetric product. */
    Symm,

    /** Whole symmetric rank k update. */
    Syrk,

    /** Whole symmetric rank two k update. */
    Syr2k,

    /** Whole triangular matrix product. */
    Trmm,

    /** Whole triangular matrix substitution. */
    Trsm,
}

/** Built-in arithmetic sources. Missing optional runtimes are represented by null. */
public object EngineBackends {
    /** Portable coverage for every operation without resolving a host library. */
    public val portable: EngineBackend get() = portableBackend

    /** Owned JVM Vector API arithmetic, or null when unavailable on this runtime. */
    public val simd: EngineBackend? get() = simdBackend()

    /** Adapts an already opened binding without changing the default engine. */
    public fun vendor(binding: Blas): EngineBackend {
        val vectors = binding as? DenseVectorKernels ?: BindingVectorKernels(binding)
        return EngineBackend(
            binding.vendor.vendorName.lowercase(),
            vectors,
            binding = binding,
            vectorName = vectors.name,
            minimumVectorLength = 1,
        )
    }
}

internal expect fun simdBackend(): EngineBackend?

/** Shared schedules and portable interface defaults provide the intrinsic availability floor. */
internal val portableBackend: EngineBackend = EngineBackend(
    "portable",
    implementations = EngineImplementation.entries.toSet(),
    panels = PortablePanelKernels,
    products = PortableProductKernels,
    triangular = PortableTriangularKernels,
    vectorName = "scalar",
)
