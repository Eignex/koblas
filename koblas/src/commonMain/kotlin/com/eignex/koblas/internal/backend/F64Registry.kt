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
 * it. [Seam] and the ranking are shared, so what differs between two registries is only their role types.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class F64Registry {
    private val vectorKernelSeam = Seam<F64Kernels>(::recompose)
    private val blasSeam = Seam<F64Blas>(::recompose)
    private val decompositionsSeam = Seam<F64Decompositions>(::recompose)
    private val sparseVectorKernelSeam = Seam<F64SparseKernels>(::recompose)
    private val sparseBlasSeam = Seam<F64SparseBlas>(::recompose)
    private val sparseDecompositionsSeam = Seam<F64SparseDecompositions>(::recompose)
    private val generalSparseLuSeam = Seam<F64GeneralSparseLu>(::recompose)
    private val repeatedSparseLuSeam = Seam<F64RepeatedSparseLu>(::recompose)
    private val sparseCholeskySeam = Seam<F64SparseCholesky>(::recompose)
    private val sparseLdlSeam = Seam<F64SparseLdl>(::recompose)
    private val sparseQrSeam = Seam<F64SparseQr>(::recompose)
    private val basisFactorizationsSeam = Seam<F64BasisFactorizations>(::recompose)
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
     * seam for, or nothing but halves [only] left out.
     *
     * [only] narrows the offer to those halves, for a caller that will take some of what a backend carries
     * and not the rest. Null offers every half, which is what a caller with no reason to choose wants.
     */
    fun offer(backend: Backend, explicit: Boolean, only: Set<BackendSlot>? = null): Boolean {
        // Counted rather than short-circuited: an object implementing several halves is offered to all of them.
        var offered = 0
        for ((slot, half) in halves) {
            if (only != null && slot !in only) continue
            if (half.offer(backend, explicit)) offered++
        }
        if (only == null || BackendSlot.F64SparseDecompositions in only) {
            if (offerSparseCapabilities(backend, explicit)) offered++
        }
        return offered > 0
    }

    private fun offerSparseCapabilities(backend: Backend, explicit: Boolean): Boolean {
        // A backend offers the roles it implements. Filling the wide seam alone no longer offers anything:
        // the roles are the contract, and every binding here implements the ones it can answer.
        var offered = false
        val general = backend
            .takeUnless { it is F64RepeatedSparseLu || it is F64BasisFactorizations } as? F64GeneralSparseLu
        val cholesky = backend as? F64SparseCholesky
        val ldl = backend as? F64SparseLdl
        val qr = backend as? F64SparseQr
        general?.let {
            generalSparseLuSeam.register(it, explicit)
            offered = true
        }
        (backend as? F64RepeatedSparseLu)?.let {
            repeatedSparseLuSeam.register(it, explicit)
            offered = true
        }
        cholesky?.let {
            sparseCholeskySeam.register(it, explicit)
            offered = true
        }
        ldl?.let {
            sparseLdlSeam.register(it, explicit)
            offered = true
        }
        qr?.let {
            sparseQrSeam.register(it, explicit)
            offered = true
        }
        (backend as? F64BasisFactorizations)?.let {
            basisFactorizationsSeam.register(it, explicit)
            offered = true
        }
        return offered
    }

    /** The general sparse LU registered under [name], or null when nothing did. */
    fun generalSparseLuNamed(name: String): F64GeneralSparseLu? = generalSparseLuSeam.named(name)

    /** The repeated-pattern sparse LU registered under [name], or null when nothing did. */
    fun repeatedSparseLuNamed(name: String): F64RepeatedSparseLu? = repeatedSparseLuSeam.named(name)

    /** The sparse Cholesky provider registered under [name], or null when nothing did. */
    fun sparseCholeskyNamed(name: String): F64SparseCholesky? = sparseCholeskySeam.named(name)

    /** The sparse LDL provider registered under [name], or null when nothing did. */
    fun sparseLdlNamed(name: String): F64SparseLdl? = sparseLdlSeam.named(name)

    /** The sparse QR provider registered under [name], or null when nothing did. */
    fun sparseQrNamed(name: String): F64SparseQr? = sparseQrSeam.named(name)

    /** The basis factorization provider registered under [name], or null when nothing did. */
    fun basisFactorizationsNamed(name: String): F64BasisFactorizations? = basisFactorizationsSeam.named(name)

    /** The basis solvers registered under [name], or null when nothing did. */
    fun basisSolversNamed(name: String): F64BasisSolvers? = basisSolverSeam.named(name)

    /** The names registered for [slot], strongest first. */
    fun namesFor(slot: BackendSlot): List<String> = halves.getValue(slot).names

    /** Names offered for a public semantic [role], strongest first. */
    fun namesFor(role: com.eignex.koblas.BackendRole): List<String> = when (role) {
        com.eignex.koblas.BackendRole.SPARSE_GENERAL_LU -> generalSparseLuSeam.all.map { it.name }
        com.eignex.koblas.BackendRole.SPARSE_REPEATED_LU -> repeatedSparseLuSeam.all.map { it.name }
        com.eignex.koblas.BackendRole.SPARSE_CHOLESKY -> sparseCholeskySeam.all.map { it.name }
        com.eignex.koblas.BackendRole.SPARSE_LDL -> sparseLdlSeam.all.map { it.name }
        com.eignex.koblas.BackendRole.SPARSE_QR -> sparseQrSeam.all.map { it.name }
        com.eignex.koblas.BackendRole.BASIS_FACTORIZATIONS -> basisFactorizationsSeam.all.map { it.name }
        else -> namesFor(role.legacySlot)
    }

    /** Clears the override and every registration, leaving the portable fallbacks. */
    fun reset() {
        installed = null
        halves.values.forEach { it.seam.reset() }
        generalSparseLuSeam.reset()
        repeatedSparseLuSeam.reset()
        sparseCholeskySeam.reset()
        sparseLdlSeam.reset()
        sparseQrSeam.reset()
        basisFactorizationsSeam.reset()
    }

    /** Builds a context from the currently registered halves, falling back to the portable reference. */
    private fun assemble(): F64Context {
        val reference = F64ReferenceSparseLinearAlgebra
        val general = generalSparseLuSeam.active ?: reference
        val cholesky = sparseCholeskySeam.active ?: reference
        val ldl = sparseLdlSeam.active ?: reference
        val qr = sparseQrSeam.active ?: reference
        val uniform = general === cholesky && general === ldl && general === qr
        val roles = if (uniform && general is F64SparseDecompositions) {
            general
        } else {
            F64SparseDecompositionRoles(general, cholesky, ldl, qr)
        }
        return F64Context(
            kernels = F64RoutedKernels(vectorKernelSeam.active),
            blas = blasSeam.active ?: F64ReferenceLinearAlgebra,
            decompositions = decompositionsSeam.active ?: F64ReferenceLinearAlgebra,
            sparseKernels = sparseVectorKernelSeam.active ?: F64PlatformSparseKernels,
            sparseBlas = sparseBlasSeam.active ?: F64ReferenceSparseLinearAlgebra,
            sparseDecompositions = roles,
            basisSolvers = basisSolverSeam.active ?: F64ReferenceSparseLinearAlgebra,
            dispatchPolicy = com.eignex.koblas.F64DispatchPolicy.AUTO,
            fallbackPolicy = com.eignex.koblas.F64FallbackPolicy.ALLOW,
            fallbackWarning = {},
            generalSparseLu = general,
            repeatedSparseLu = repeatedSparseLuSeam.active,
            sparseCholesky = cholesky,
            sparseLdl = ldl,
            basisFactorizations = basisFactorizationsSeam.active ?: reference,
        )
    }

    private fun recompose() {
        changes.incrementAndFetch()
    }
}

private val com.eignex.koblas.BackendRole.legacySlot: BackendSlot
    get() = when (this) {
        com.eignex.koblas.BackendRole.DENSE_KERNELS -> BackendSlot.F64Kernels
        com.eignex.koblas.BackendRole.DENSE_BLAS -> BackendSlot.F64Blas
        com.eignex.koblas.BackendRole.DENSE_DECOMPOSITIONS -> BackendSlot.F64Decompositions
        com.eignex.koblas.BackendRole.SPARSE_KERNELS -> BackendSlot.F64SparseKernels
        com.eignex.koblas.BackendRole.SPARSE_BLAS -> BackendSlot.F64SparseBlas
        com.eignex.koblas.BackendRole.SPARSE_DECOMPOSITIONS -> BackendSlot.F64SparseDecompositions
        com.eignex.koblas.BackendRole.BASIS_SOLVERS -> BackendSlot.F64BasisSolvers
        else -> error("$this has a semantic seam")
    }
