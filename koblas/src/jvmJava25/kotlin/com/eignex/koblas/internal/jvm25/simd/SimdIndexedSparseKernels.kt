package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.sparse.DispatchingIndexedSparseKernels
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.internal.sparse.SparseTuning
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/** JVM indexed kernels: the vector dense dot where [SparseSimd] measured it faster, scalar everywhere else. */
internal object SimdIndexedSparseKernels : DispatchingIndexedSparseKernels() {
    override val name: String = "simd"

    override fun select(operation: SparseOperation, count: Int): IndexedSparseKernels =
        if (usesVector(operation, count)) SparseSimd else ScalarIndexedSparseKernels

    /**
     * Whether the dot reaches its vector body. The lane width bounds the configured crossover rather than being
     * covered by it: a deployment may set the crossover to one, and a support narrower than a lane block leaves
     * every entry to the scalar tail, which is scalar work whatever entered it.
     */
    private fun usesVector(operation: SparseOperation, count: Int): Boolean =
        operation == SparseOperation.DotDense && SparseSimd.vectorDot &&
            count >= SparseTuning.simdIndexedCrossover && count >= SparseSimd.lanes
}
