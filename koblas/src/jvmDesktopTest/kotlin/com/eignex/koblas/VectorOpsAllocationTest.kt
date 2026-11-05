package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.*
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/**
 * Vector operations over views of one buffer read their operands in place: neither a disjoint interleaving
 * nor an identical mapping takes a snapshot, and a sparse read of a strided view gathers nothing.
 */
@RunWith(Parameterized::class)
class VectorOpsAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(
            listOf(
                with(SharedBufferFixture()) {
                    AllocationCase("disjoint shared buffer vector copy allocates nothing", 1_000) {
                        destination.copyFrom(source)
                        backing
                    }
                },
                with(SharedBufferFixture()) {
                    AllocationCase("disjoint shared buffer vector axpy allocates nothing", 1_000) {
                        destination.axpy(1e-12, source)
                        backing
                    }
                },

                with(OverlappingFixture()) {
                    AllocationCase("overlapping contiguous copy allocates nothing", 1_000) {
                        destination.copyFrom(source)
                        backing
                    }
                },
            ) + identicalMappingCases() + sparseViewCases(),
        )

        // Copy returns on an identical mapping before anything depends on the stride, so one
        // contiguous case suffices; axpy reads the operand through its stride, so it also runs reversed.
        private fun identicalMappingCases(): List<AllocationCase> = listOf(
            with(IdenticalMappingFixture(1)) {
                AllocationCase("identical contiguous vector copy allocates nothing", 1_000) {
                    destination.copyFrom(source)
                    backing
                }
            },

        ) + intArrayOf(1, -3).map { stride ->
            with(IdenticalMappingFixture(stride)) {
                AllocationCase("identical vector axpy at stride $stride allocates nothing", 1_000) {
                    destination.axpy(1e-12, source)
                    backing
                }
            }
        }

        private fun sparseViewCases(): List<AllocationCase> = intArrayOf(-2, -1, 1, 2).map { stride ->
            with(SparseViewFixture(stride)) {
                AllocationCase("sparse dot against a dense view at stride $stride allocates nothing", 1_000) {
                    sparse dot view
                }
            }
        }
    }

    /** One logical vector reached through two wrappers, so only the mapping, not the object, is shared. */
    private class IdenticalMappingFixture(stride: Int) {
        val n = 512
        val backing = DoubleArray(3 * n + 4) { it * 0.01 }
        val offset = if (stride > 0) 2 else 2 + (n - 1) * -stride
        val source = StridedVector(backing, offset, n, stride)
        val destination = source.view(0, source.size)
    }

    private class SharedBufferFixture {
        val n = 512
        val backing = DoubleArray(2 * n) { it * 0.01 }
        val source = StridedVector(backing, 0, n, 2)
        val destination = StridedVector(backing, 1, n, 2)
    }

    private class OverlappingFixture {
        val n = 512
        val backing = DoubleArray(n + 1) { it * 0.01 }
        val source = StridedVector(backing, 0, n)
        val destination = StridedVector(backing, 1, n)
    }

    private class SparseViewFixture(stride: Int) {
        val n = 512
        val backing = DoubleArray(2 * n) { it * 0.01 }
        val sparse = SparseVector.wrap(n, intArrayOf(0, 7, n - 1), doubleArrayOf(1.5, 0.0, -2.0))
        val view = StridedVector(backing, if (stride < 0) 2 * n - 2 else 1, n, stride)
    }
}
