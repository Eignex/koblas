@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.configuration.*
import com.eignex.koblas.internal.dense.DensePanelKernels
import com.eignex.koblas.internal.dense.DenseProductKernels
import com.eignex.koblas.internal.dense.DenseTriangularKernels
import com.eignex.koblas.internal.dense.PortableDenseBlas
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.PortableProductKernels
import com.eignex.koblas.internal.dense.PortableTriangularKernels
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.host.DenseHostSelection
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.PortableSparseBlas
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.internal.sparse.SparseKernelAdapter
import com.eignex.koblas.internal.sparse.SparsePanelKernels
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.vendor.Blas

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine by lazy { platformEngine() }

/** Platform kernels with eligible whole-call host composition and portable fallbacks. */
internal expect fun platformEngine(): KoblasEngine

/**
 * Immutable dense and sparse BLAS with portable fallbacks.
 *
 * [koblas] is the platform default. [portable] creates an explicitly portable engine for application code.
 * [BuiltinEngines] provides exact portable and JVM SIMD compositions
 * for tests and benchmarks, neither of which loads a host library. The platform default may compose
 * [vendor] into eligible calls.
 */
public class KoblasEngine private constructor(
    /** Contiguous and strided dense Level 1 kernels. */
    public val vectorKernels: DenseVectorKernels,
    internal val indexedSparseKernels: IndexedSparseKernels,
    /** Optional production native adapter, exposed only for expert benchmark attribution. */
    public val vendor: Blas?,
    /** Panel arithmetic and the local execution grouping that dense and sparse algorithms schedule with. */
    internal val panelKernels: DensePanelKernels,
    /** Register-tile product arithmetic and the tile geometry dense products pack for. */
    internal val productKernels: DenseProductKernels,
    /** Diagonal-block substitution and the right-hand-side grouping triangular routines schedule with. */
    internal val triangularKernels: DenseTriangularKernels,
    private val dense: DenseBlas,
    private val sparseBlas: SparseBlas,
) : DenseBlas by dense,
    SparseBlas by sparseBlas {
    /** Execution policy and arithmetic components have one composition owner. */
    internal constructor(
        vectorKernels: DenseVectorKernels,
        indexedSparseKernels: IndexedSparseKernels,
        vendor: Blas?,
        panelKernels: DensePanelKernels = PortablePanelKernels,
        productKernels: DenseProductKernels = PortableProductKernels,
        triangularKernels: DenseTriangularKernels = PortableTriangularKernels,
        hostSelection: DenseHostSelection? = null,
    ) : this(
        vectorKernels,
        indexedSparseKernels,
        vendor,
        panelKernels,
        productKernels,
        triangularKernels,
        PortableDenseBlas(
            vectorKernels,
            panelKernels,
            productKernels,
            triangularKernels,
            hostSelection,
        ),
        PortableSparseBlas(vectorKernels, indexedSparseKernels, SparsePanelKernels(vectorKernels, panelKernels)),
    )

    /**
     * Allocates a product using this engine. Two sparse operands produce owned CSC storage;
     * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
     */
    public fun gemm(a: Matrix, b: Matrix): Matrix = product(1.0, a, false, b, false)

    /** Scaled product with independent transpose flags and the storage rule of [gemm]. */
    public fun gemm(alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean): Matrix =
        product(alpha, a, transposeA, b, transposeB)

    /**
     * Writes a generic product through this engine into addressed dense storage. Validation precedes all
     * writes; zero alpha and empty depth read no inputs, and zero beta reads no destination values.
     * [workspace] lends logical alias snapshots, sparse scratch and custom operand normalization.
     */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun gemm(
        alpha: Double,
        a: Matrix,
        transposeA: Boolean,
        b: Matrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace? = null,
    ): Unit = productInto(alpha, a, transposeA, b, transposeB, beta, c, workspace)

    override fun gemm(a: DenseMatrixRegion, b: DenseMatrixRegion): DenseMatrix =
        product(1.0, a, false, b, false) as DenseMatrix

    override fun gemm(a: SparseMatrix, b: DenseMatrixRegion): DenseMatrix =
        product(1.0, a, false, b, false) as DenseMatrix

    override fun gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix = product(1.0, a, false, b, false) as SparseMatrix

    /** TRSV over readable dense, sparse or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trsv(
        a: Matrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        workspace: Workspace? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        val sparse = storedCsc(a)
        if (sparse != null) {
            trsv(sparse, x, lower, transpose, unitDiag, workspace)
        } else {
            triangleOperand(
                workspace,
                a,
                lower,
                unitDiag,
            ) { dense -> trsv(dense, x, lower, transpose, unitDiag, workspace) }
        }
    }

    /** TRMV over readable dense, sparse or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trmv(
        a: Matrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        workspace: Workspace? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        val sparse = storedCsc(a)
        if (sparse != null) {
            trmv(sparse, x, lower, transpose, unitDiag, workspace)
        } else {
            triangleOperand(
                workspace,
                a,
                lower,
                unitDiag,
            ) { dense -> trmv(dense, x, lower, transpose, unitDiag, workspace) }
        }
    }

    /** TRSM over readable dense, sparse or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trsm(
        a: Matrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trsm")
        if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
            com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
            return
        }
        val sparse = storedCsc(a)
        if (sparse != null) {
            trsm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
        } else {
            triangleOperand(
                workspace,
                a,
                lower,
                unitDiag,
            ) { dense -> trsm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
        }
    }

    /** TRMM over readable dense, sparse or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trmm(
        a: Matrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trmm")
        if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
            com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
            return
        }
        val sparse = storedCsc(a)
        if (sparse != null) {
            trmm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
        } else {
            triangleOperand(
                workspace,
                a,
                lower,
                unitDiag,
            ) { dense -> trmm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
        }
    }

    /** Explicit application engine selection without changing the platform default. */
    public companion object {
        /**
         * Combines available arithmetic with ordered routing preferences and complete portable coverage.
         *
         * Backends named by no applicable rule contribute no arithmetic. Missing backends fall through to
         * later preferences and then portable Kotlin. This never changes [koblas]. The caller chooses an
         * applicable profile explicitly; automatic platform selection also checks [EngineProfile.appliesTo].
         * Panel, product and triangular families are selected unconditionally before a schedule starts,
         * because packing and grouping must use the same geometry as the selected arithmetic.
         * Whole-call portable preferences retain the shared schedule with those separately selected families.
         */
        public fun combine(profile: EngineProfile, vararg backends: EngineBackend): KoblasEngine {
            require(backends.map { it.name }.distinct().size == backends.size) { "Duplicate backend identifiers" }
            val families = setOf(
                EngineImplementation.Panels,
                EngineImplementation.Products,
                EngineImplementation.Triangular,
            )
            require(profile.rules.none { rule -> !rule.unconditional && rule.implementations.any { it in families } }) {
                "Scheduled kernel families require unconditional selection before packing"
            }
            val routing = CompiledRouting(profile, backends.toList())
            val acceleratedVectors = EngineImplementation.entries.takeWhile { it != EngineImplementation.IndexedDot }
                .any(routing::hasPreference)
            val vectors = if (acceleratedVectors) CombinedVectorKernels(routing, profile.name) else ScalarVectorKernels
            val indexed = if (routing.hasPreference(EngineImplementation.IndexedDot)) {
                CombinedIndexedKernels(routing)
            } else {
                ScalarIndexedSparseKernels
            }
            val binding = backends.filterIsInstance<BackendContributions>().firstNotNullOfOrNull { it.binding }
            val hostSelection = DenseHostSelection { operation, rows, columns, depth, borrowed ->
                routing.select(operation, rows, columns, depth, borrowed = borrowed).binding
            }
            return KoblasEngine(
                vectors,
                indexed,
                binding,
                routing.select(EngineImplementation.Panels).panels!!,
                routing.select(EngineImplementation.Products).products!!,
                routing.select(EngineImplementation.Triangular).triangular!!,
                hostSelection = hostSelection.takeIf { binding != null },
            )
        }

        /**
         * Creates an immutable engine using portable Kotlin for every dense and sparse operation.
         *
         * Does not resolve a host library or require JVM modules or native-access flags. Keep the returned
         * instance for repeated calls; each concurrent invocation needs its own destinations and workspace.
         * Creating it does not change [koblas] or the engines used by container convenience operations.
         *
         * Arithmetic follows routine-specific BLAS semantics. Implementations may fuse operations and
         * reassociate reductions; rounding and unspecified exceptional results can vary between backends.
         */
        public fun portable(): KoblasEngine = KoblasEngine(ScalarVectorKernels, ScalarIndexedSparseKernels, null)
    }

    /** Sparse-vector kernels composed from this engine's dense and indexed Level 1 selections. */
    public val sparseKernels: SparseKernels = SparseKernelAdapter(
        indexedSparseKernels.name,
        vectorKernels,
        indexedSparseKernels,
    )

    /**
     * Short read-only description of what this engine selected, for logs.
     *
     * A selection, not a claim about execution: which of the selected implementations a given call reaches
     * depends on its shape.
     */
    public val name: String
        get() = "${vectorKernels.name}/${sparseKernels.name}/${panelKernels.name}/${productKernels.name}/" +
            triangularKernels.name

    override fun toString(): String = "KoblasEngine($name)"
}

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * The JVM default retains [simd] when available and [scalar] otherwise as its owned kernels, and may
 * compose eligible oneMKL matrix calls when native access is enabled. Native has no SIMD
 * engine and uses an applicable measured host profile, or [scalar] when none applies.
 * Android and desktop JDKs below 25 use [scalar] without resolving a host library.
 */
public object BuiltinEngines {
    /** Portable Kotlin across all three levels. */
    @get:kotlin.jvm.JvmStatic
    public val scalar: KoblasEngine get() = scalarEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * On the JVM, [koblas] retains these kernels when available, including in its host composition.
     * A window too short or too strided for a vector body runs a portable fallback.
     */
    @get:kotlin.jvm.JvmStatic
    public val simd: KoblasEngine? by lazy {
        simdBackend()?.let { KoblasEngine.combine(requireNotNull(StandardEngineProfiles.jvmSimd), it) }
    }
}

/** Shared scalar composition; availability and default selection remain platform decisions. */
internal val scalarEngine: KoblasEngine by lazy { KoblasEngine.portable() }
