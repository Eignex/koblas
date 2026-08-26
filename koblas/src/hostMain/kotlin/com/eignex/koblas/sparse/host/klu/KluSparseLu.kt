@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64SparseLuAdapter
import kotlinx.cinterop.*

/** Sparse LU factorizations backed by a host KLU 2. */
public class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseLuAdapter(config.factorizeMin, config.equilibrate) {
    private val loader = KluLoader(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val nativeAvailable: Boolean get() = loader.available

    /**
     * Factor [a] reusing the symbolic analysis behind [previous], which supersedes it and must not be
     * solved after this call. Reuse is what KLU is for: a circuit keeps one sparsity pattern and factors it
     * again for every operating point, and the analysis is the expensive half.
     *
     * A [previous] from another backend, or one whose pattern does not match, is answered as [factor]
     * would. This is KLU's own routine rather than a seam method, since no other sparse LU koblas binds
     * carries an analysis worth reusing.
     */
    public fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "refactor")
        if (!nativeAvailable) return factor(a)
        val reusable = previous as? KluFactorization ?: return factor(a)
        if (reusable.n != a.rows) return factor(a)
        // KLU reuses the ordering it analyzed, so a pattern it was not analyzed for has to start over.
        return if (reusable.refactor(a.copyColumnPointers(), a.copyRowIndices(), a.values)) {
            reusable
        } else {
            F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
    }

    override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val functions = loader.functions ?: error("KLU 2 is not available")
        val common = loader.common(equilibrate)
        val symbolic = nativeHeap.alloc<COpaquePointerVar>().also { it.value = null }
        val numeric = nativeHeap.alloc<COpaquePointerVar>().also { it.value = null }
        val colPtr = a.copyColumnPointers()
        val rowIdx = a.copyRowIndices()

        symbolic.value = colPtr.usePinned { ap ->
            rowIdx.usePinned { ai -> functions.analyze(a.rows, ap.addressOf(0), ai.addressOf(0), common) }
        }
        if (symbolic.value == null) {
            release(functions, symbolic, numeric, common)
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }

        numeric.value = colPtr.usePinned { ap ->
            rowIdx.usePinned { ai ->
                a.values.usePinned { ax ->
                    functions.factor(ap.addressOf(0), ai.addressOf(0), ax.addressOf(0), symbolic.value, common)
                }
            }
        }
        if (numeric.value == null) {
            release(functions, symbolic, numeric, common)
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        return KluFactorization(
            KluFactorization.KluHandle(symbolic.ptr, numeric.ptr, common, functions),
            functions,
            a.rows,
        )
    }

    /** Frees what a failed factorization allocated, since no cleaner has been attached to it yet. */
    private fun release(
        functions: KluFunctions,
        symbolic: COpaquePointerVar,
        numeric: COpaquePointerVar,
        common: CPointer<ByteVar>,
    ) {
        functions.freeNumeric(numeric.ptr, common)
        functions.freeSymbolic(symbolic.ptr, common)
        nativeHeap.free(numeric)
        nativeHeap.free(symbolic)
        nativeHeap.free(common)
    }
}
