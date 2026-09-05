package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64Context
import com.eignex.koblas.F64DispatchPolicy
import com.eignex.koblas.F64FallbackPolicy
import com.eignex.koblas.SparseRoles
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64RoutedKernels
import com.eignex.koblas.sparse.F64BasisFactorizations
import com.eignex.koblas.sparse.F64GeneralSparseLu
import com.eignex.koblas.sparse.F64QuasiDefiniteLdl
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseCholesky
import com.eignex.koblas.sparse.F64SparseDecompositionRoles
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseQr
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import kotlin.concurrent.Volatile
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.concurrent.atomics.incrementAndFetch

/**
 * The double-precision seams and the context they compose. One element type's registrations live in one
 * of these, so an element type added later brings its own rather than widening this one: its halves are
 * different interfaces, and a backend offered to the public registry belongs to whichever registry recognises
 * it. [Seam] and the ranking are shared, so what differs between two registries is only their role types.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class F64Registry {
    /**
     * One seam per half, derived from [BackendSlot] rather than listed again, so a half added there arrives
     * here with its type test, its portable default and its keys already attached.
     *
     * A seam takes only what its slot accepted, which is what makes reading one back as the half's own
     * interface a cast that cannot fail.
     */
    private val halves: Map<BackendSlot, Seam<Backend>> =
        BackendSlot.entries.associateWith { Seam<Backend>(::recompose) }

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
        while (true) {
            val at = changes.load()
            resolution.load()?.let { if (it.at == at) return it.context }
            val fresh = Resolution(at, assemble())
            // Published only when no registration overtook the assembly. A backend filling several halves
            // registers into them one at a time, so an assembly that started between two of them sees it in
            // one seam and not the other; caching that would hand the same half-applied context to every
            // later reader until the next change. Re-reading the count is what separates the two cases.
            if (changes.load() == at) {
                resolution.store(fresh)
                return fresh.context
            }
        }
    }

    /** Overrides [active] wholesale; null restores automatic selection. */
    fun install(context: F64Context?) {
        installed = context
    }

    /**
     * Offers [backend] to every seam whose half it implements, and reports whether any took it. Explicit
     * offers outrank automatic ones. A caller that gets false has offered something this element type has no
     * seam for, or nothing but halves [offered] left out.
     *
     * [offered] narrows the offer, for a caller that will take some of what a backend carries and not the
     * rest, and names the halves a deployment asked this backend for. Null offers every half through the
     * specialization policy, which is what a caller with no reason to choose wants.
     */
    fun offer(backend: Backend, explicit: Boolean, offered: BackendOffer? = null): Boolean {
        // Counted rather than short-circuited: an object implementing several halves is offered to all of them.
        var taken = 0
        for ((slot, seam) in halves) {
            if (offered != null && slot !in offered.halves) continue
            // A half named for this backend was asked for, so only the type test stands between them.
            val takes =
                if (offered?.wasNamed(slot) == true) slot.accepts(backend) else slot.acceptsOffer(backend)
            if (!takes) continue
            seam.register(backend, explicit)
            taken++
        }
        return taken > 0
    }

    /** The backend registered for [slot] under [name], or null when nothing did. */
    fun named(slot: BackendSlot, name: String): Backend? = halves.getValue(slot).named(name)

    /** The names registered for [slot], strongest first. */
    fun namesFor(slot: BackendSlot): List<String> = halves.getValue(slot).all.map { it.name }

    /** Names offered for a public semantic [role], strongest first. */
    fun namesFor(role: BackendRole): List<String> = namesFor(role.slot)

    /** Clears the override and every registration, leaving the portable fallbacks. */
    fun reset() {
        installed = null
        halves.values.forEach { it.reset() }
    }

    /** Builds a context from the currently registered halves, falling back to the portable reference. */
    private fun assemble(): F64Context {
        val general = resolved<F64GeneralSparseLu>(BackendSlot.F64GeneralSparseLu)
        val cholesky = resolved<F64SparseCholesky>(BackendSlot.F64SparseCholesky)
        val quasiDefiniteLdl = resolved<F64QuasiDefiniteLdl>(BackendSlot.F64QuasiDefiniteLdl)
        val qr = resolved<F64SparseQr>(BackendSlot.F64SparseQr)
        return F64Context(
            // The routed kernels wrap whatever host registered rather than replacing the compiled-in ones,
            // so this half composes its offer instead of falling back to a portable default.
            kernels = F64RoutedKernels(strongest<F64Kernels>(BackendSlot.F64Kernels)),
            blas = resolved<F64Blas>(BackendSlot.F64Blas),
            decompositions = resolved<F64Decompositions>(BackendSlot.F64Decompositions),
            sparseKernels = resolved<F64SparseKernels>(BackendSlot.F64SparseKernels),
            sparseBlas = resolved<F64SparseBlas>(BackendSlot.F64SparseBlas),
            sparseDecompositions = F64SparseDecompositionRoles(general, cholesky, quasiDefiniteLdl, qr),
            basisSolvers = resolved<F64BasisSolvers>(BackendSlot.F64BasisSolvers),
            dispatchPolicy = F64DispatchPolicy.AUTO,
            fallbackPolicy = F64FallbackPolicy.ALLOW,
            fallbackWarning = {},
            roles = SparseRoles(
                generalLu = general,
                repeatedLu = strongest<F64RepeatedSparseLu>(BackendSlot.F64RepeatedSparseLu),
                cholesky = cholesky,
                quasiDefiniteLdl = quasiDefiniteLdl,
                qr = qr,
                basisFactorizations = resolved<F64BasisFactorizations>(BackendSlot.F64BasisFactorizations),
            ),
        )
    }

    /** The strongest offer for [slot], or null when nothing registered one. */
    private inline fun <reified T : Backend> strongest(slot: BackendSlot): T? = halves.getValue(slot).active as? T

    /**
     * The strongest offer for [slot], or koblas's own implementation of the half when there is none.
     *
     * Only for a half [BackendSlot.required] marks: an optional half stands in a placeholder its own
     * interface does not accept, so the error names the slot rather than arriving as a bare cast failure.
     */
    private inline fun <reified T : Backend> resolved(slot: BackendSlot): T = strongest<T>(slot)
        ?: slot.portableDefault() as? T
        ?: error("${slot.name} has no portable default of its own half's type")

    private fun recompose() {
        changes.incrementAndFetch()
    }
}
