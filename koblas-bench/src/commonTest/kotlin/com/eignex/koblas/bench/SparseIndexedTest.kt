package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import kotlin.test.*

class SparseIndexedTest {
    @Test
    fun `indexed workloads agree with the scalar reference on repeated calls`() {
        for (operation in listOf("spdot", "spdot-raw", "spdot-sparse", "spaxpy", "spaxpy-raw",
            "spnrm2", "spasum", "spscatter", "spscatter-raw", "spgather", "spgather-zero")) {
            val case = Cases.parse("$operation+65+sparse-uniform+density=0.25").single()
            val reference = assertNotNull(sparseIndexedArm(case, KoblasEngine.scalar).work)
            for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd)) {
                val work = assertNotNull(sparseIndexedArm(case, engine).work)

                repeat(3) {
                    DenseReference.check(doubleArrayOf(reference.run()), doubleArrayOf(work.run()), case.id)
                }
            }
        }
    }

    @Test
    fun `indexed preflight rejects a finite incorrect reduction`() {
        val broken = object : EngineKernels by KoblasEngine.scalar.kernels {
            override fun dot(indices: IntArray, indexOffset: Int, values: DoubleArray, valueOffset: Int,
                count: Int, dense: DoubleArray): Double = 0.0
        }
        val case = Cases.parse("spdot-raw+65+sparse-uniform+density=0.25").single()

        assertFailsWith<IllegalStateException> { sparseIndexedArm(case, broken) }
    }

    @Test
    fun `indexed preflight checks mutations beyond the returned sink`() {
        val scalar = KoblasEngine.scalar.kernels
        val broken = object : EngineKernels by scalar {
            override fun gather(indices: IntArray, indexOffset: Int, values: DoubleArray, valueOffset: Int,
                count: Int, source: DoubleArray) {
                scalar.gather(indices, indexOffset, values, valueOffset, count, source)
                values[valueOffset + 1] += 1.0
            }
        }
        val case = Cases.parse("spgather+65+sparse-uniform+density=0.25").single()

        assertFailsWith<IllegalStateException> { sparseIndexedArm(case, broken) }
    }

    @Test
    fun `indexed preflight checks zeroed dense positions beyond the returned sink`() {
        val scalar = KoblasEngine.scalar.kernels
        val broken = object : EngineKernels by scalar {
            override fun gatherZero(indices: IntArray, indexOffset: Int, values: DoubleArray, valueOffset: Int,
                count: Int, source: DoubleArray) {
                scalar.gatherZero(indices, indexOffset, values, valueOffset, count, source)
                source[indices[indexOffset + 1]] = 1.0
            }
        }
        val case = Cases.parse("spgather-zero+65+sparse-uniform+density=0.25").single()

        assertFailsWith<IllegalStateException> { sparseIndexedArm(case, broken) }
    }
}
