@file:OptIn(com.eignex.koblas.UnsafeKoblasApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.*

/** Indexed workloads check their complete buffers against the scalar engine before timing. */
internal fun sparseIndexedArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    val size = case.dimension(0)
    val density = case.option("density", "0.01").toDouble()
    val x = Fixtures.sparseVector(size, density, 1)
    val sparseY = Fixtures.sparseVector(size, density, 2)
    val initialDense = Fixtures.vector(size, 2)
    val dense = initialDense.copyOf()
    val operation = case.operation
    val mutation = operation in setOf("spaxpy", "spaxpy-raw", "spscatter", "spscatter-raw", "spgather", "spgather-zero")
    val timing = if (mutation) case.option("timing", "reset-and-arithmetic") else "arithmetic"

    fun arithmetic(selected: KoblasEngine, sparse: SparseVector, destination: DoubleArray): () -> Double {
        val kernels = selected.kernels
        val indices = sparse.copyIndices()
        val initialValues = sparse.values.copyOf()
        val count = sparse.nnz
        return when (operation) {
            "spdot" -> { { kernels.dot(sparse.indices, 0, sparse.values, 0, count, destination) } }
            "spdot-raw" -> { { kernels.dot(indices, 0, sparse.values, 0, count, destination) } }
            "spdot-sparse" -> { { kernels.dot(sparse.indices, 0, sparse.values, 0, count,
                sparseY.indices, 0, sparseY.values, 0, sparseY.nnz) } }
            "spnrm2" -> { { kernels.normVector(sparse) } }
            "spasum" -> { { kernels.absoluteSumVector(sparse) } }
            "spaxpy" -> { {
                initialDense.copyInto(destination)
                kernels.axpy(sparse.indices, 0, sparse.values, 0, count, 0.875, destination)
                destination[0]
            } }
            "spaxpy-raw" -> { {
                initialDense.copyInto(destination)
                kernels.axpy(indices, 0, sparse.values, 0, count, 0.875, destination)
                destination[0]
            } }
            "spscatter" -> { {
                initialValues.copyInto(sparse.values); initialDense.copyInto(destination)
                kernels.scatter(sparse.indices, 0, sparse.values, 0, count, destination)
                sparse.values[0]
            } }
            "spscatter-raw" -> { {
                initialDense.copyInto(destination)
                kernels.scatter(indices, 0, sparse.values, 0, count, destination)
                destination[0]
            } }
            "spgather" -> { {
                if (timing == "reset-and-arithmetic") initialDense.copyInto(destination)
                kernels.gather(indices, 0, sparse.values, 0, count, destination)
                sparse.values[0] + sparse.values[count - 1]
            } }
            "spgather-zero" -> { {
                initialValues.copyInto(sparse.values); initialDense.copyInto(destination)
                kernels.gatherZero(indices, 0, sparse.values, 0, count, destination)
                sparse.values[0]
            } }
            else -> error("Unsupported indexed workload $operation")
        }
    }

    val reference = SparseVector.of(size, x.copyIndices(), x.values.copyOf())
    val referenceDense = initialDense.copyOf()
    val expected = arithmetic(KoblasEngine.scalar, reference, referenceDense)()
    val run = arithmetic(engine, x, dense)
    DenseReference.check(doubleArrayOf(expected), doubleArrayOf(run()), case.id)
    DenseReference.check(reference.values, x.values, "${case.id} sparse backing storage")
    DenseReference.check(referenceDense, dense, "${case.id} dense backing storage")
    initialDense.copyInto(dense)
    return ArmChoice(engineWork(engine, operation, timing, run,
        if (operation == "spgather") x.values else null), null)
}
