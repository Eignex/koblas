@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseHostEligibility
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseMatrixRoute
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseProductKernels
import com.eignex.koblas.dense.DenseTriangularKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.HostDenseBlas
import com.eignex.koblas.dense.PortableDenseBlas
import com.eignex.koblas.dense.PortablePanelKernels
import com.eignex.koblas.dense.PortableProductKernels
import com.eignex.koblas.dense.PortableTriangularKernels
import com.eignex.koblas.dense.RoutedDenseBlas
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.PortableSparseBlas
import com.eignex.koblas.sparse.SPARSE_SCHEDULING
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernelAdapter
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparsePanelKernels
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.openBlas

/**
 * The immutable platform-selected BLAS engine used by top-level convenience operations.
 *
 * Deferred rather than computed while this file's class initializes. Picking an engine reads
 * [selectedVendor], which is declared below and is itself deferred; resolving eagerly would read that
 * property's backing delegate before the initializer reached it and select against a null vendor.
 */
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
 * [vendor] into eligible calls. The [routeOf] overloads describe which implementations a call reaches.
 */
public class KoblasEngine internal constructor(
    /** Contiguous and strided dense Level 1 kernels. */
    public val vectorKernels: DenseVectorKernels,
    internal val indexedSparseKernels: IndexedSparseKernels,
    /** Optional production native adapter, exposed only for expert benchmark attribution. */
    public val vendor: Blas?,
    /** Panel arithmetic and the local execution grouping that dense and sparse algorithms schedule with. */
    internal val panelKernels: DensePanelKernels = PortablePanelKernels,
    /** Register-tile product arithmetic and the tile geometry dense products pack for. */
    internal val productKernels: DenseProductKernels = PortableProductKernels,
    /** Diagonal-block substitution and the right-hand-side grouping triangular routines schedule with. */
    internal val triangularKernels: DenseTriangularKernels = PortableTriangularKernels,
    sparseKernelName: String = indexedSparseKernels.name,
    nativeEligible: DenseHostEligibility? = null,
    private val dense: RoutedDenseBlas =
        PortableDenseBlas(vectorKernels, panelKernels, productKernels, triangularKernels).let { portable ->
            vendor?.let { HostDenseBlas(portable, it, requireNotNull(nativeEligible)) } ?: portable
        },
    private val sparseBlas: SparseBlas = PortableSparseBlas(
        vectorKernels,
        indexedSparseKernels,
        SparsePanelKernels(vectorKernels, panelKernels),
    ),
) : DenseBlas by dense,
    SparseBlas by sparseBlas {
    /**
     * Allocates a product using this engine. Two sparse or prepared operands produce owned CSC storage;
     * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
     * Prepared snapshots contribute data and cached orientation, never execution policy.
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
        alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean,
        beta: Double, c: DenseMatrixRegion, workspace: Workspace? = null,
    ): Unit = productInto(alpha, a, transposeA, b, transposeB, beta, c, workspace)

    override fun gemm(a: DenseMatrixRegion, b: DenseMatrixRegion): DenseMatrix =
        product(1.0, a, false, b, false) as DenseMatrix

    override fun gemm(a: SparseMatrix, b: DenseMatrixRegion): DenseMatrix =
        product(1.0, a, false, b, false) as DenseMatrix

    override fun gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix =
        product(1.0, a, false, b, false) as SparseMatrix

    /** TRSV over readable dense, sparse, prepared or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trsv(a: Matrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false,
        workspace: Workspace? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        val sparse = storedCsc(a)
        if (sparse != null) trsv(sparse, x, lower, transpose, unitDiag, workspace)
        else triangleOperand(workspace, a, lower, unitDiag) { dense -> trsv(dense, x, lower, transpose, unitDiag, workspace) }
    }

    /** TRMV over readable dense, sparse, prepared or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trmv(a: Matrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false,
        workspace: Workspace? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        val sparse = storedCsc(a)
        if (sparse != null) trmv(sparse, x, lower, transpose, unitDiag, workspace)
        else triangleOperand(workspace, a, lower, unitDiag) { dense -> trmv(dense, x, lower, transpose, unitDiag, workspace) }
    }

    /** TRSM over readable dense, sparse, prepared or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trsm(a: Matrix, b: DenseMatrixRegion, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false,
        right: Boolean = false, alpha: Double = 1.0,
        workspace: Workspace? = null,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trsm")
        if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
            com.eignex.koblas.dense.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
            return
        }
        val sparse = storedCsc(a)
        if (sparse != null) trsm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
        else triangleOperand(workspace, a, lower, unitDiag) { dense -> trsm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
    }

    /** TRMM over readable dense, sparse, prepared or custom operands, using this engine. */
    @Suppress("LongParameterList") // triangle flags and workspace
    public fun trmm(a: Matrix, b: DenseMatrixRegion, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false,
        right: Boolean = false, alpha: Double = 1.0,
        workspace: Workspace? = null,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trmm")
        if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
            com.eignex.koblas.dense.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
            return
        }
        val sparse = storedCsc(a)
        if (sparse != null) trmm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
        else triangleOperand(workspace, a, lower, unitDiag) { dense -> trmm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
    }

    /** GEMV over an engine-neutral prepared snapshot. */
    @Suppress("LongParameterList") // BLAS operands and workspace
    public fun gemv(alpha: Double, a: PreparedSparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray,
        transpose: Boolean = false, workspace: Workspace? = null,
    ): Unit = gemv(alpha, a.snapshot, x, beta, y, transpose, workspace)

    /** SYMV over an engine-neutral prepared snapshot. */
    @Suppress("LongParameterList") // BLAS operands and workspace
    public fun symv(alpha: Double, a: PreparedSparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray,
        lower: Boolean = true, workspace: Workspace? = null,
    ): Unit = symv(alpha, a.snapshot, x, beta, y, lower, workspace)

    /** Explicit application engine selection without changing the platform default. */
    public companion object {
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
    public val sparseKernels: SparseKernels = SparseKernelAdapter(sparseKernelName, vectorKernels, indexedSparseKernels)

    /**
     * The component that owns built-in sparse Level 2 and 3 calls.
     *
     * Always this library's portable CSC scheduling. The Level 1 kernels an individual column reaches are a
     * separate question, and [com.eignex.koblas.sparse.SparseBlas.routeOf] is what answers it.
     */
    public val sparseImplementation: String = SPARSE_SCHEDULING

    /**
     * Short read-only description of what this engine selected, for logs.
     *
     * A selection, not a claim about execution: which of the selected implementations a given call reaches
     * depends on its shape. The [routeOf] overloads describe concrete dense and sparse calls.
     */
    public val name: String
        get() = "${vectorKernels.name}/${sparseKernels.name}/${panelKernels.name}/${productKernels.name}/" +
            triangularKernels.name

    /**
     * The route for a dense matrix [operation] with the supplied [call] facts.
     *
     * Names the panels and kernels reached by the shared schedule, including short-window fallbacks.
     * A whole-call host route identifies the library and entry point.
     */
    public fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute =
        dense.routeOf(operation, call)

    /**
     * The route for a dense vector call of this [operation] and [length].
     *
     * [contiguous] says whether the operands have unit stride. A composed route means values
     * decide which implementation finishes the call; inspecting a route does not execute it.
     */
    public fun routeOf(
        operation: DenseOperation,
        length: Int,
        contiguous: Boolean = true,
    ): VectorRoute<DenseOperation> = vectorRoute(
        operation,
        operation.name.lowercase(),
        length,
        vectorKernels.name,
        vectorKernels.implementationFor(operation, length, contiguous),
    )

    /** The route for a sparse vector call over [count] stored entries. */
    public fun routeOf(operation: SparseOperation, count: Int): VectorRoute<SparseOperation> =
        sparseKernels.routeOf(operation, count)

    override fun toString(): String = "KoblasEngine($name)"
}

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * The JVM default retains [simd] when available and [scalar] otherwise as its owned kernels, and may
 * compose eligible oneMKL matrix calls when native access is enabled. Native has no SIMD
 * engine and composes an installed host library with portable fallbacks, or uses [scalar] without one.
 * Android and desktop JDKs below 25 use [scalar] without resolving a host library.
 */
public expect object BuiltinEngines {
    /** Portable Kotlin across all three levels. */
    public val scalar: KoblasEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * On the JVM, [koblas] retains these kernels when available, including in its host composition.
     * A window too short or too strided for a vector body runs a portable fallback. [KoblasEngine.routeOf]
     * identifies the bodies reached by a particular call rather than treating the whole engine as vectorized.
     */
    public val simd: KoblasEngine?
}

/** Optional platform-default Native host binding, resolved once only when that platform policy asks for it. */
internal val selectedVendor: Blas? by lazy { openBlas() }

/** Shared scalar composition; availability and default selection remain platform decisions. */
internal val scalarEngine: KoblasEngine by lazy { KoblasEngine.portable() }
