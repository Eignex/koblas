@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.basicReport
import com.eignex.koblas.sparse.host.applyEquilibration
import com.eignex.koblas.sparse.internal.replaceColumns
import com.eignex.koblas.sparse.internal.snapshot
import com.eignex.koblas.sparse.requireSolveShapes
import com.eignex.koblas.withColumn
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.math.abs
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** A host BASICLU factorization with deterministic close and cleaner fallback for its native object. */
public open class BasicluFactorization internal constructor(
    internal val handle: BasicluObjectHandle,
    internal val functions: BasicluFunctions,
    override val n: Int,
    /** The equilibration the values were scaled by before factorization, or null when there was none. */
    private val rowScale: DoubleArray? = null,
) : F64SparseLuFactorization {

    /** The object and the call that frees it, its own class so the cleaner captures it and not this. */
    internal class BasicluObjectHandle(val obj: CPointer<ByteVar>, private val functions: BasicluFunctions) {
        fun release() {
            functions.free(obj)
            nativeHeap.free(obj)
        }
    }

    private val lifecycle = NativeResourceLifecycle("BASICLU factorization", handle::release)

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(lifecycle) { it.close() }
    private val scratchSolveAllocation = AllocationCapability(
        AllocationGuarantee.NO_MANAGED,
        listOf(ScratchRequirement(ScratchKind.F64, n)),
    )

    override val failedAt: Int get() = NOT_SINGULAR

    override val nnz: Int get() = anchoring {
        (basicluStatistic(handle.obj, BasicluStore.LNZ) + basicluStatistic(handle.obj, BasicluStore.UNZ)).toInt()
    }

    override val rcond: Double get() = anchoring {
        val largest = abs(basicluStatistic(handle.obj, BasicluStore.MAX_PIVOT))
        if (largest == 0.0) 0.0 else abs(basicluStatistic(handle.obj, BasicluStore.MIN_PIVOT)) / largest
    }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability {
        val needsRhs = aliasing || (rowScale != null && !transpose)
        return if (needsRhs) scratchSolveAllocation else noManagedAllocation
    }

    override fun report(): F64SparseFactorizationReport = basicReport("basiclu")

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireSolveShapes(n, b, out)
        // BASICLU reads the right-hand side and writes the destination, and a forward solve scales what goes
        // in, neither of which may touch the caller's array.
        if (out === b || (rowScale != null && !transpose)) {
            return workspace.borrow(n) { rhs ->
                b.copyInto(rhs)
                if (rowScale != null && !transpose) applyEquilibration(rhs, rowScale)
                solveDistinct(rhs, out, transpose)
            }
        }
        return solveDistinct(b, out, transpose)
    }

    private fun solveDistinct(rhs: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        val flag = if (transpose) BASICLU_TRANSPOSED else BASICLU_FORWARD
        val status = anchoring {
            rhs.usePinned { source ->
                out.usePinned { destination ->
                    functions.solveDense(handle.obj, source.addressOf(0), destination.addressOf(0), flag)
                }
            }
        }
        check(status == BasicluStatus.OK) { "basiclu_obj_solve_dense failed with status $status" }
        if (rowScale != null && transpose) applyEquilibration(out, rowScale)
        return out
    }

    override fun close(): Unit = lifecycle.close()

    /**
     * Runs [body] with this factorization reachable from a global, so the cleaner cannot free the object
     * while the native call inside it is reading it.
     */
    internal fun <R> anchoring(body: () -> R): R = lifecycle.withResource {
        val previous = AnchoredBasiclu.held
        AnchoredBasiclu.held = this
        try {
            body()
        } finally {
            AnchoredBasiclu.held = previous
        }
    }
}

/** A host BASICLU basis whose factors follow a column replacement until BASICLU declines the update. */
public class BasicluBasisFactorization internal constructor(
    private val owner: BasicluSparseLu,
    initialBasis: F64SparseMatrix,
    handle: BasicluObjectHandle,
    functions: BasicluFunctions,
) : BasicluFactorization(handle, functions, initialBasis.rows),
    F64BasisFactorization {
    private var built: F64SparseMatrix = initialBasis
    private val pending = LinkedHashMap<Int, F64SparseVector>()

    /**
     * The basis these factors stand for, built from the replacements BASICLU took when it is asked for and
     * not before. Building it copies the structure of the whole matrix and validates it again, which is
     * work per pivot that a driver pivoting to optimality asks for once, if at all.
     */
    override val basis: F64SparseMatrix
        get() {
            if (pending.isNotEmpty()) {
                built = replaceColumns(built, pending)
                pending.clear()
            }
            return built
        }

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        // BASICLU wants the entering column solved for and then the leaving row named through a transposed
        // solve, both before the update itself.
        val status = anchoring {
            val prepared = entering.indices.map(Int::toLong).toLongArray().usePinned { rows ->
                entering.values.usePinned { values ->
                    functions.solveForUpdate(
                        handle.obj,
                        entering.indices.size.toLong(),
                        rows.addressOf(0),
                        values.addressOf(0),
                        BASICLU_FORWARD,
                        0L,
                    )
                }
            }
            if (prepared != BasicluStatus.OK) {
                prepared
            } else {
                val named = longArrayOf(column.toLong()).usePinned { leaving ->
                    functions.solveForUpdate(handle.obj, 1L, leaving.addressOf(0), null, BASICLU_TRANSPOSED, 0L)
                }
                if (named != BasicluStatus.OK) named else functions.update(handle.obj, 0.0)
            }
        }
        // Kept as a copy, since the caller's vector stays live for it to write and the build comes later.
        pending[column] = entering.snapshot()
        if (status != BasicluStatus.OK) {
            val replacement = owner.factorBasis(basis)
            close()
            return replacement
        }
        return this
    }
}

/** What [BasicluSparseLu.factorBasis] answers with for a basis BASICLU called singular. */
internal class BasicluSingularBasisFactorization(
    private val owner: BasicluSparseLu,
    override val basis: F64SparseMatrix,
) : F64BasisFactorization {
    override val n: Int get() = basis.rows
    override val failedAt: Int get() = com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
    override val nnz: Int get() = 0
    override val rcond: Double get() = 0.0

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization =
        owner.factorBasis(basis.withColumn(column, entering))

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw SingularMatrix(failedAt, "solve: the factorization is singular")
}

/** Holds the factorization a native call is reading, the counterpart of UMFPACK's own anchor. */
@ThreadLocal
internal object AnchoredBasiclu {
    var held: BasicluFactorization? = null
}
