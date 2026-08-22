package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.Workspace
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorization
import java.lang.foreign.MemorySegment
import java.lang.ref.Cleaner
import java.lang.ref.Reference

private val cleaner: Cleaner = Cleaner.create()

/** A SuperLU factor handle, freed when its Kotlin owner becomes unreachable. */
public class SuperluFactorization internal constructor(
    override val n: Int,
    override val failedAt: Int,
    private val handle: MemorySegment,
    override val nnz: Int,
) : F64SparseFactorization {
    init {
        val held = handle
        cleaner.register(this) { SuperluCalls.free(held) }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        if (out !== b) b.copyInto(out)
        try {
            check(SuperluCalls.solve(handle, MemorySegment.ofArray(out), transpose) == 0) { "SuperLU solve failed" }
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }

    override fun determinant(): Double = try {
        SuperluCalls.determinant(handle)
    } finally {
        Reference.reachabilityFence(this)
    }
}
