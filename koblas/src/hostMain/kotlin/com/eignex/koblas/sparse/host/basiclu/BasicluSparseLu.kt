@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseLuAdapter
import com.eignex.koblas.sparse.host.equilibrationScale
import com.eignex.koblas.sparse.host.scaledValues
import kotlinx.cinterop.*

/**
 * Sparse LU and Forrest-Tomlin basis updates backed by a host BASICLU.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one. What a
 * caller reaches this backend by name for is the routines it carries outside the seam, and a wrapper
 * would hide their type. Only the name and the ranking are meant to vary.
 */
public open class BasicluSparseLu(
    /** Policy for this backend instance. */
    public val config: BasicluConfig = BasicluConfig(),
) : F64SparseLuAdapter(config.factorizeMin, config.equilibrate) {
    private val loader = BasicluLoader(config)

    override val name: String get() = BackendNames.BASICLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
    final override val nativeAvailable: Boolean get() = loader.available

    /**
     * Whether [factorBasis] answers with a factorization that updates its factors in place. False without
     * the library, where a replacement costs a factorization and a caller pacing its own refactorizations
     * has nothing left to pace.
     */
    public val supportsBasisUpdates: Boolean get() = nativeAvailable

    /**
     * BASICLU offers no row scaling of its own, so equilibration is applied to the values handed over and
     * undone in the solves, by the same power-of-two factors the portable factorization uses.
     */
    final override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val rowIdx = a.copyRowIndices()
        val scale = if (equilibrate) equilibrationScale(a.rows, rowIdx, a.values) else null
        val values = if (scale == null) a.values else scaledValues(rowIdx, a.values, scale)
        val functions = loader.functions ?: error("BASICLU is not available")
        val handle = factored(a, rowIdx, values, functions)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return BasicluFactorization(handle, functions, a.rows, scale)
    }

    /**
     * Factor a simplex [basis] for column replacements, which may bring in any column at all.
     *
     * BASICLU's own routine rather than a seam method: it carries the basis independently of any matrix,
     * which is what lets an entering column be arbitrary, and no other sparse LU koblas binds does that.
     * A basis is factorized natively at any size, since what a caller wants from BASICLU is the update that
     * follows and the gate the general factorization answers to has nothing to say about it.
     */
    public fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        val functions = loader.functions.takeIf { nativeAvailable }
            ?: return F64RefactoringBasisFactorization(this, basis, factor(basis))
        val handle = factored(basis, basis.copyRowIndices(), basis.values, functions)
            ?: return BasicluSingularBasisFactorization(this, basis)
        return BasicluBasisFactorization(this, basis, handle, functions)
    }

    /** An initialized object carrying the factors of [a]'s pattern with [values], or null if it would not. */
    private fun factored(
        a: F64SparseMatrix,
        rowIdx: IntArray,
        values: DoubleArray,
        functions: BasicluFunctions,
    ): BasicluFactorization.BasicluObjectHandle? {
        val obj = allocateBasicluObject()
        if (functions.initialize(obj, a.rows.toLong()) != BasicluStatus.OK) {
            functions.free(obj)
            nativeHeap.free(obj)
            return null
        }
        val colPtr = a.copyColumnPointers()
        val status = LongArray(a.cols) { colPtr[it].toLong() }.usePinned { begin ->
            LongArray(a.cols) { colPtr[it + 1].toLong() }.usePinned { end ->
                LongArray(rowIdx.size) { rowIdx[it].toLong() }.usePinned { rows ->
                    values.usePinned { entries ->
                        functions.factorize(
                            obj,
                            begin.addressOf(0),
                            end.addressOf(0),
                            rows.addressOf(0),
                            entries.addressOf(0),
                        )
                    }
                }
            }
        }
        if (status == BasicluStatus.OK) return BasicluFactorization.BasicluObjectHandle(obj, functions)
        functions.free(obj)
        nativeHeap.free(obj)
        return null
    }
}
