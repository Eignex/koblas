package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.ComposedSeam
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Seam
import com.eignex.koblas.mathBackend

// Which dense backend is active, and how it got there. The sparse counterpart is `sparse/Registry.kt`, and
// the two are deliberately the same file in two packages: same seams, same verbs, same fallback rule.
//
// The state machine itself lives in `Seam` and `ComposedSeam`. What is left here is the part that is
// genuinely dense-specific — that a `Blas` and a `Lapack` compose into a `LinearAlgebra` by delegation, and
// that this side has a platform-discovery hook.

/**
 * Registers whatever backends this platform provides.
 *
 * The single discovery hook. The JVM's actual scans the classpath with `ServiceLoader` and registers its
 * host-OpenBLAS halves; the native targets register eagerly before `main` instead (see the cblas
 * AutoInstall), which is earlier than this would run, so theirs is a no-op; the web targets have nothing
 * to reach. Whatever it finds goes through [registerBlas] / [registerLapack] / [registerVectorKernels] like any
 * other backend, so there is one ranking and one fallback rather than a parallel path.
 */
internal expect fun registerPlatformBackends()

private val blasSeam = Seam<Blas>(::recompose)

private val lapackSeam = Seam<Lapack>(::recompose)

/**
 * The dense level-1 seam.
 *
 * Internal, and read directly by `Primitives.kt` rather than through an accessor, because every `dot` in
 * the library goes through it. Its [Seam.active] is nullable and null means the compiled-in kernel — see
 * [VectorKernels] for why this one half cannot have an object fallback.
 */
internal val vectorKernelSeam = Seam<VectorKernels>()

private val denseSeam = ComposedSeam<LinearAlgebra>(
    default = ReferenceLinearAlgebra,
    onFirstRead = lazy { registerPlatformBackends() },
)

/** The active backend: an [installLinearAlgebra] override when set, else the strongest registered half of
 *  each kind, else the portable [ReferenceLinearAlgebra]. */
val koblas: LinearAlgebra get() = denseSeam.active

/**
 * Pairs a [Blas] and a [Lapack] into one backend.
 *
 * Only used when the two halves come from different places; when one backend won both, that object is
 * used directly, so the common case keeps a single virtual call rather than two.
 */
private class ComposedBackend(private val blas: Blas, private val lapack: Lapack) :
    LinearAlgebra,
    Blas by blas,
    Lapack by lapack {
    override val name: String get() = if (blas.name == lapack.name) blas.name else "${blas.name}+${lapack.name}"
    override val priority: Int get() = maxOf(blas.priority, lapack.priority)
}

/** Recomputes the composed backend from the two halves, falling back to the reference for either. */
private fun recompose() {
    val blas = blasSeam.active
    val lapack = lapackSeam.active
    denseSeam.resolve(
        when {
            blas == null && lapack == null -> null
            blas === lapack && blas is LinearAlgebra -> blas
            else -> ComposedBackend(blas ?: ReferenceLinearAlgebra, lapack ?: ReferenceLinearAlgebra)
        },
    )
}

/** What this runtime resolved on both performance seams, for startup logging — e.g.
 *  `"backend=openblas, primitives=simd(8 lanes)"` ([koblas].name and [mathBackend]). */
val koblasInfo: String get() = "backend=${koblas.name}, primitives=$mathBackend"

/**
 * Overrides the backend [koblas] resolves to, taking precedence over every automatic mechanism —
 * registration and platform discovery alike. Passing null restores automatic selection. The switch
 * is visible to subsequent [koblas] reads but is not synchronized with operations in flight.
 */
fun installLinearAlgebra(backend: LinearAlgebra?) {
    denseSeam.install(backend)
}

/**
 * Overrides which [VectorKernels] the primitives use, taking precedence over [registerVectorKernels]. Passing null
 * restores automatic selection, and clearing it restores the built-in kernels.
 *
 * The counterpart of [installLinearAlgebra] for this half. Not thread-safe against concurrent level-1
 * calls: install during startup, before other threads run.
 */
fun installVectorKernels(backend: VectorKernels?) {
    vectorKernelSeam.install(backend)
}

/**
 * Offers [backend] for automatic selection: it becomes active only while no [installLinearAlgebra]
 * override is set and no previously registered backend has a higher [Backend.priority]. This
 * is how backend artifacts activate themselves on platforms without classpath discovery — the
 * koblas-cblas startup registration goes through here — and it keeps the strongest of several
 * candidates: openblas over cblas over the reference.
 */
fun registerLinearAlgebra(backend: LinearAlgebra) {
    registerBlas(backend)
    registerLapack(backend)
}

/**
 * Offers the BLAS half of [backend] for automatic selection, ranked by [Backend.priority] against other
 * registered BLAS backends. Register the halves separately when a host provides one library and not the
 * other; [registerLinearAlgebra] is the shorthand for registering the same object as both.
 */
fun registerBlas(backend: Blas) {
    blasSeam.register(backend)
}

/** Offers the LAPACK half of [backend] for automatic selection; see [registerBlas]. */
fun registerLapack(backend: Lapack) {
    lapackSeam.register(backend)
}

/**
 * Offers [backend] for automatic selection as the level-1 kernels, ranked by [Backend.priority]. This is
 * how koblas activates the host kernels on the targets that can reach a host BLAS; see [registerBlas] for
 * the same mechanism on the other halves.
 */
fun registerVectorKernels(backend: VectorKernels) {
    vectorKernelSeam.register(backend)
}

/** Test hook: clears the matrix-seam override and registration so selection tests are order-independent. */
internal fun resetRegisteredLinearAlgebra() {
    denseSeam.reset()
    blasSeam.reset()
    lapackSeam.reset()
}

/** Test hook: clears the level-1 override and registration; see [resetRegisteredLinearAlgebra]. */
internal fun resetRegisteredVectorKernels() {
    vectorKernelSeam.reset()
}

/** LU-factorize this square matrix with the active backend ([koblas]). */
fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(this, b, transpose)

/** Matrix-matrix product `this · other` with the active backend. */
fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
