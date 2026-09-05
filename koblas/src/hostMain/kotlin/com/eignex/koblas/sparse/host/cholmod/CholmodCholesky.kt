@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import kotlinx.cinterop.*

/**
 * CHOLMOD's sparse Cholesky, as the routine a SuiteSparse backend fills its Cholesky half with.
 *
 * Not a backend itself. CHOLMOD has no LU, and a backend offering only half of what the seam carries would
 * take the seam from one that offers all of it; the SuiteSparse bindings hold this instead, so the
 * collection answers both routines natively however the seam resolves.
 */
public actual class CholmodCholesky actual constructor(config: CholmodConfig) {
    private val loader = CholmodLoader(config)

    /** Whether libcholmod opened and every symbol this needs resolved. */
    public actual val isAvailable: Boolean get() = loader.available

    /**
     * Factor [a]'s lower triangle, or null when the library is unusable, which lets the caller fall back
     * rather than fail.
     *
     * @throws NotPositiveDefinite at the column CHOLMOD stopped at, matching the portable Cholesky.
     */
    public actual fun factor(a: F64SparseMatrix): F64SparseCholeskyFactorization? = build(a, CHOLMOD_TRUE) { minor, n ->
        if (minor < n) {
            throw NotPositiveDefinite(minor, 0.0, "cholesky: CHOLMOD stopped at column $minor, which is not positive")
        }
        NOT_SINGULAR
    }?.let(::CholmodCholeskyFactorization)

    /**
     * Factor [a]'s lower triangle into `L·D·Lᵀ`, or null when the library is unusable.
     *
     * A zero pivot comes back as a factorization reporting `singular` at its column rather than raising,
     * since an `L·D·Lᵀ` failing means the matrix is singular where an `L·Lᵀ` failing only means it was not
     * positive definite.
     */
    public actual fun factorQuasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization? =
        build(a, finalLl = 0) { minor, n -> if (minor < n) minor else NOT_SINGULAR }
            ?.let(::CholmodLdlFactorization)

    private inline fun build(
        a: F64SparseMatrix,
        finalLl: Int,
        verdict: (minor: Int, n: Int) -> Int,
    ): CholmodFactorization? {
        val functions = loader.functions ?: return null
        val common = loader.common(finalLl)
        val sparse = describeLowerTriangle(a)
        // The factorize return separates a matrix CHOLMOD could not factor from one it factored and found
        // indefinite: a non-positive pivot still returns TRUE and shows up in `minor`, where running out of
        // memory returns FALSE over a pattern-only factor that `minor` reports as complete. Discarded, the
        // second came back as a successful factorization.
        var factored = true
        val factor = try {
            functions.analyze(sparse, common).also { analyzed ->
                if (analyzed != null) {
                    factored = functions.factorize(sparse, analyzed.reinterpret(), common) == CHOLMOD_TRUE
                }
            }
        } finally {
            freeMatrix(sparse)
        }
        if (factor != null && !factored) {
            releaseFactor(functions, factor, common)
            return null
        }
        if (factor == null) {
            functions.finish(common)
            nativeHeap.free(common)
            return null
        }
        val block = NativeBlock(factor.reinterpret())
        val n = block.getSize(CHOLMOD_FACTOR_N).toInt()
        val minor = block.getSize(CHOLMOD_FACTOR_MINOR).toInt()
        val handle = CholmodFactorization.CholmodHandle(factor, common, functions, n)
        val failedAt = try {
            verdict(minor, n)
        } catch (raised: NotPositiveDefinite) {
            handle.release()
            throw raised
        }
        return CholmodFactorization(handle, functions, n, failedAt)
    }
}

/** Frees a factor CHOLMOD produced but could not finish, along with the common it was built against. */
private fun releaseFactor(functions: CholmodFunctions, factor: COpaquePointer, common: CPointer<ByteVar>) {
    memScoped {
        val slot = alloc<COpaquePointerVar>()
        slot.value = factor
        functions.freeFactor(slot.ptr, common)
    }
    functions.finish(common)
    nativeHeap.free(common)
}
