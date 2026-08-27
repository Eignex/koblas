package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.factorization.cholesky.F64SparseUpLookingCholesky
import kotlin.test.*

class SparseAllocationContractTest {

    @Test
    fun `strict portable lu requires all scratch before mutation`() {
        val factor = F64ReferenceSparseLinearAlgebra.factor(matrix())
        val b = doubleArrayOf(2.0, 8.0)
        val out = doubleArrayOf(17.0, 19.0)
        val workspace = Workspace().apply { reserve(2, count = 1) }

        val failure = assertFailsWith<AllocationPolicyRejectedException> {
            factor.solveInto(
                b,
                out,
                workspace = workspace,
                allocationPolicy = AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE,
            )
        }

        assertEquals(AllocationGuarantee.NO_MANAGED_OR_NATIVE, failure.capability.guarantee)
        assertContentEquals(doubleArrayOf(17.0, 19.0), out)
    }

    @Test
    fun `reserved portable lu honors strongest allocation policy`() {
        val factor = F64ReferenceSparseLinearAlgebra.factor(matrix())
        val workspace = Workspace().apply { reserve(2, count = 2) }
        val out = DoubleArray(2)

        factor.solveInto(
            doubleArrayOf(2.0, 8.0),
            out,
            workspace = workspace,
            allocationPolicy = AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE,
        )

        assertContentEquals(doubleArrayOf(0.0, 2.0), out)
        assertEquals(2, workspace.available(ScratchRequirement(ScratchKind.F64, 2)))
    }

    @Test
    fun `allocation free sparse cholesky needs no workspace`() {
        val factor = F64SparseUpLookingCholesky.factorLower(matrix())
        val out = DoubleArray(2)

        factor.solveInto(
            doubleArrayOf(2.0, 8.0),
            out,
            allocationPolicy = AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE,
        )

        assertContentEquals(doubleArrayOf(0.0, 2.0), out)
    }

    @Test
    fun `undeclared third party guarantee is rejected before invocation`() {
        var called = false
        val factor = object : F64SparseFactorization {
            override val n: Int get() = 1
            override val failedAt: Int get() = NOT_SINGULAR
            override val nnz: Int get() = 1
            override val rcond: Double get() = 1.0
            override fun solveInto(
                b: DoubleArray,
                out: DoubleArray,
                transpose: Boolean,
                workspace: Workspace?,
            ): DoubleArray {
                called = true
                return out
            }
        }
        val out = doubleArrayOf(5.0)

        assertFailsWith<AllocationPolicyRejectedException> {
            factor.solveInto(
                doubleArrayOf(1.0),
                out,
                allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
            )
        }

        assertFalse(called)
        assertContentEquals(doubleArrayOf(5.0), out)
    }

    private fun matrix(): F64SparseMatrix = F64SparseMatrix.ofTriplets(
        2,
        2,
        intArrayOf(0, 1, 0, 1),
        intArrayOf(0, 0, 1, 1),
        doubleArrayOf(2.0, 1.0, 1.0, 4.0),
    )
}
