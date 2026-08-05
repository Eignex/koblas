package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLinearAlgebra
import com.eignex.koblas.sparse.SparseVectorKernels

/**
 * Every backend koblas will use for a piece of work, in one object you can hold.
 *
 * [koblas] is the process-wide default, assembled from whatever registered itself, and the free functions
 * use it — so nothing has to know contexts exist. Build your own when you want the choice to be explicit
 * rather than ambient: a test that must not depend on what the platform happened to install, a benchmark
 * comparing two backends in the same process, a solver instance pinned to a configuration, a run someone
 * needs to reproduce.
 *
 * A `LinearAlgebra` and a `SparseLinearAlgebra` at once, by delegation, so a context *is* a backend and
 * `context.gemv(...)` works wherever `koblas.gemv(...)` does. That is also what let the old
 * `ComposedBackend` and `ComposedSparseBackend` classes go: pairing two halves into a whole was their
 * entire job, and a context already does it.
 *
 * Six halves, and the split between them is the one the rest of the library uses: three dense and three
 * sparse, vector kernels below the matrix routines below the factorizations. Construct by name — six
 * positional backends would be unreadable — or, more usually, adjust the default with [with]:
 *
 * ```kotlin
 * val portable = koblas.with(blas = ReferenceLinearAlgebra, lapack = ReferenceLinearAlgebra)
 * val x = portable.solve(portable.factor(a), b)
 * ```
 *
 * Immutable, so it is safe to share between threads and cheap to keep: the fields are final, which is
 * strictly better than the global it replaces for the hot path, since a `final` kernel reference lets a
 * null check hoist out of a loop where a `@Volatile` one cannot.
 *
 * @property vectorKernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property lapack dense factorizations.
 * @property sparseVectorKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property sparseLapack sparse factorizations.
 */
class KoblasContext(
    val vectorKernels: VectorKernels,
    val blas: Blas,
    val lapack: Lapack,
    val sparseVectorKernels: SparseVectorKernels,
    val sparseBlas: SparseBlas,
    val sparseLapack: SparseLapack,
) : LinearAlgebra,
    Blas by blas,
    Lapack by lapack,
    SparseLinearAlgebra,
    SparseBlas by sparseBlas,
    SparseLapack by sparseLapack {

    /**
     * The distinct names of the backends that do the matrix work, joined — e.g. `"openblas"` when one
     * library won everything, or `"openblas+reference"` when the sparse halves fell back.
     *
     * The two vector-kernel halves are deliberately left out. They are reported by [mathBackend], and
     * folding them in here made every name carry `"simd(4 lanes)+"` in front of the answer anyone was
     * actually asking for. [koblasInfo] prints both parts.
     *
     * Explicit rather than delegated because every half is a [Backend] and Kotlin cannot pick one to inherit
     * this from. Deduplicated in encounter order, since the usual case is one or two real answers repeated.
     */
    override val name: String
        get() = listOf(blas, lapack, sparseBlas, sparseLapack).map { it.name }.distinct().joinToString("+")

    /** The strongest half's priority: a context is at least as preferred as the best thing in it. */
    override val priority: Int
        get() = maxOf(
            vectorKernels.priority,
            blas.priority,
            lapack.priority,
            sparseVectorKernels.priority,
            sparseBlas.priority,
            sparseLapack.priority,
        )

    /**
     * A copy with the named halves replaced and the rest kept.
     *
     * The ergonomic way to build a context, because it starts from one that already resolved: the six-way
     * constructor makes you name a backend for every half, which is the right default for a type you can
     * install but tedious when you want to change one thing.
     */
    fun with(
        vectorKernels: VectorKernels = this.vectorKernels,
        blas: Blas = this.blas,
        lapack: Lapack = this.lapack,
        sparseVectorKernels: SparseVectorKernels = this.sparseVectorKernels,
        sparseBlas: SparseBlas = this.sparseBlas,
        sparseLapack: SparseLapack = this.sparseLapack,
    ): KoblasContext = KoblasContext(
        vectorKernels = vectorKernels,
        blas = blas,
        lapack = lapack,
        sparseVectorKernels = sparseVectorKernels,
        sparseBlas = sparseBlas,
        sparseLapack = sparseLapack,
    )

    override fun toString(): String = "KoblasContext($name)"
}
