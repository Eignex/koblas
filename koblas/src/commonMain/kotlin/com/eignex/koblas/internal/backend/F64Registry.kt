package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64Context
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import kotlin.concurrent.Volatile
import kotlin.concurrent.atomics.*

/**
 * The double-precision seams and the context they compose. One element type's registrations live in one
 * of these, so an element type added later brings its own rather than widening this one: its halves are
 * different interfaces, and a backend offered to the public registry belongs to whichever registry recognises
 * it. [Seam] and the ranking are shared, so what differs between two registries is only the six types.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class F64Registry {
    private val vectorKernelSeam = Seam<F64Kernels>(::recompose)
    private val blasSeam = Seam<F64Blas>(::recompose)
    private val decompositionsSeam = Seam<F64Decompositions>(::recompose)
    private val sparseVectorKernelSeam = Seam<F64SparseKernels>(::recompose)
    private val sparseBlasSeam = Seam<F64SparseBlas>(::recompose)
    private val sparseDecompositionsSeam = Seam<F64SparseDecompositions>(::recompose)
    private val basisSolverSeam = Seam<F64BasisSolvers>(::recompose)

    /**
     * One seam paired with the cast that recognises its half. The cast is written out rather than reflected,
     * since [BackendSlot] names an interface and only the compiler can turn that name into a type test.
     */
    private class Half<T : Backend>(val seam: Seam<T>, private val cast: (Backend) -> T?) {
        /** Offers [backend] to this seam if it implements this half, and reports whether it did. */
        fun offer(backend: Backend, explicit: Boolean): Boolean {
            val half = cast(backend) ?: return false
            seam.register(half, explicit)
            return true
        }

        /** The names offered for this half, strongest first. */
        val names: List<String> get() = seam.all.map { it.name }
    }

    /**
     * Every half this registry has a seam for, by slot. Keyed rather than listed so a [BackendSlot] added
     * without a seam beside it fails here, at construction, rather than by registering nothing at all.
     */
    private val halves: Map<BackendSlot, Half<*>> = mapOf(
        BackendSlot.F64Kernels to Half(vectorKernelSeam) { it as? F64Kernels },
        BackendSlot.F64Blas to Half(blasSeam) { it as? F64Blas },
        BackendSlot.F64Decompositions to Half(decompositionsSeam) { it as? F64Decompositions },
        BackendSlot.F64SparseKernels to Half(sparseVectorKernelSeam) { it as? F64SparseKernels },
        BackendSlot.F64SparseBlas to Half(sparseBlasSeam) { it as? F64SparseBlas },
        BackendSlot.F64SparseDecompositions to Half(sparseDecompositionsSeam) { it as? F64SparseDecompositions },
        BackendSlot.F64BasisSolvers to Half(basisSolverSeam) { it as? F64BasisSolvers },
    )

    init {
        check(halves.keys == BackendSlot.entries.toSet()) {
            "every BackendSlot needs a seam: ${BackendSlot.entries - halves.keys} have none"
        }
    }

    @Volatile
    private var installed: F64Context? = null

    /** A context and the [changes] count it was assembled at, published as one so neither can be read alone. */
    private class Resolution(val at: Int, val context: F64Context)

    /** Bumped by every seam change, so a stale [resolution] can be told from a current one. */
    private val changes = AtomicInt(0)

    private val resolution = AtomicReference<Resolution?>(null)

    /**
     * The context in force: an [install] override when set, else what the seams resolved to.
     *
     * Assembled on demand against a change count rather than eagerly on every registration. Eager rebuilds
     * race: two threads registering into different halves could each assemble from a partial view, and the
     * later write would drop the other's registration with nothing left to trigger a rebuild. Reading the
     * count before assembling means a stamp that still matches cannot have missed a change, and a stamp
     * that no longer matches costs a rebuild rather than a wrong answer.
     */
    val active: F64Context get() {
        installed?.let { return it }
        val at = changes.load()
        resolution.load()?.let { if (it.at == at) return it.context }
        val fresh = Resolution(at, assemble())
        resolution.store(fresh)
        return fresh.context
    }

    /** Overrides [active] wholesale; null restores automatic selection. */
    fun install(context: F64Context?) {
        installed = context
    }

    /**
     * Offers [backend] to every seam whose half it implements, and reports whether any took it. Explicit
     * offers outrank automatic ones. A caller that gets false has offered something this element type has no
     * seam for.
     */
    fun offer(backend: Backend, explicit: Boolean): Boolean {
        // Counted rather than short-circuited: an object implementing several halves is offered to all of them.
        var offered = 0
        for (half in halves.values) if (half.offer(backend, explicit)) offered++
        return offered > 0
    }

    /** The sparse LU registered under [name], or null when nothing did. */
    fun sparseDecompositionsNamed(name: String): F64SparseDecompositions? = sparseDecompositionsSeam.named(name)

    /** The basis solvers registered under [name], or null when nothing did. */
    fun basisSolversNamed(name: String): F64BasisSolvers? = basisSolverSeam.named(name)

    /** The names registered for [slot], strongest first. */
    fun namesFor(slot: BackendSlot): List<String> = halves.getValue(slot).names

    /** Clears the override and every registration, leaving the portable fallbacks. */
    fun reset() {
        installed = null
        halves.values.forEach { it.seam.reset() }
    }

    /** Builds a context from the currently registered halves, falling back to the portable reference. */
    private fun assemble(): F64Context = F64Context(
        kernels = F64RoutedKernels(vectorKernelSeam.active),
        blas = blasSeam.active ?: F64ReferenceLinearAlgebra,
        decompositions = decompositionsSeam.active ?: F64ReferenceLinearAlgebra,
        sparseKernels = sparseVectorKernelSeam.active ?: F64PlatformSparseKernels,
        sparseBlas = sparseBlasSeam.active ?: F64ReferenceSparseLinearAlgebra,
        sparseDecompositions = sparseDecompositionsSeam.active ?: F64ReferenceSparseLinearAlgebra,
        basisSolvers = basisSolverSeam.active ?: F64ReferenceSparseLinearAlgebra,
    )

    private fun recompose() {
        changes.incrementAndFetch()
    }
}
