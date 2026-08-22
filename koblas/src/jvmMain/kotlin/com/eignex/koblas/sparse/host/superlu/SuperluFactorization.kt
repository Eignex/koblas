package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.Workspace
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorization
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.ref.Cleaner
import java.lang.ref.Reference

private val cleaner: Cleaner = Cleaner.create()

/** A SuperLU 7 factorization, retaining its native matrices and permutations in a shared arena. */
public class SuperluFactorization internal constructor(
    override val n: Int,
    override val failedAt: Int,
    private val factor: SuperluFactor,
    private val arena: Arena,
    override val nnz: Int,
) : F64SparseFactorization {
    init {
        val heldFactor = factor
        val heldArena = arena
        cleaner.register(this) {
            SuperluCalls.free(heldFactor)
            heldArena.close()
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        if (out !== b) b.copyInto(out)
        try {
            SuperluCalls.scaleRhs(factor, out, transpose)
            SuperluCalls.solve(
                factor,
                MemorySegment.ofArray(out),
                transpose,
            )
        } finally {
            Reference.reachabilityFence(this)
        }
        SuperluCalls.unscaleSolution(factor, out, transpose)
        return out
    }

    override fun determinant(): Double = try {
        SuperluCalls.determinant(
            factor,
        )
    } finally {
        Reference.reachabilityFence(this)
    }
}
