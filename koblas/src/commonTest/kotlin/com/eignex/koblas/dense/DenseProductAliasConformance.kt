package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.assertEquals

/** Alias layouts distinguish shared wrappers, shared storage and independent operand values. */
private enum class ProductAlias {
    SameMatrix,
    SharedArray,
    Left,
    Right,
    Neither,
    SharedInputs,
}

/** Checks pre-call values against the oracle, counting staging loans beyond the schedule's own scratch. */
private fun assertProductAliasesAgreeWithReference(
    call: (DenseMatrix, DenseMatrix, DenseMatrix, Workspace?) -> Unit,
    reference: (DenseMatrix, DenseMatrix, DenseMatrix) -> Unit,
) {
    val rng = Random(20261013)
    val order = 6
    // Symmetric diagonal expansion can borrow the same length as a snapshot, so measure unaliased scratch.
    val scratch = Workspace()
    call(randomMatrix(order, order, rng), randomMatrix(order, order, rng), randomMatrix(order, order, rng), scratch)
    val scratchLoans = scratch.available(order * order)
    for (alias in ProductAlias.entries) {
        for (useWorkspace in booleanArrayOf(false, true)) {
            val c = randomMatrix(order, order, rng)
            val a = when (alias) {
                ProductAlias.SameMatrix, ProductAlias.SharedArray, ProductAlias.Left -> c
                else -> randomMatrix(order, order, rng)
            }
            val b = when (alias) {
                ProductAlias.SameMatrix, ProductAlias.Right -> c
                ProductAlias.SharedArray -> DenseMatrix.wrap(order, order, c.values)
                ProductAlias.SharedInputs -> DenseMatrix.wrap(order, order, a.values)
                else -> randomMatrix(order, order, rng)
            }
            val originalA = a.copyOf()
            val originalB = b.copyOf()
            val expected = c.copyOf()
            reference(originalA, originalB, expected)
            val workspace = if (useWorkspace) Workspace() else null

            call(a, b, c, workspace)

            assertClose(expected.values, c.values, "product alias=$alias workspace=$useWorkspace")
            if (a.values !== c.values) assertClose(originalA.values, a.values, "left input changed")
            if (b.values !== c.values) assertClose(originalB.values, b.values, "right input changed")
            workspace?.let {
                val snapshots = if (a.values === c.values || b.values === c.values) 1 else 0
                assertEquals(
                    scratchLoans + snapshots,
                    it.available(order * order),
                    "product alias=$alias staging loans",
                )
            }
        }
    }
}

/** Shared storage must retain distinct shapes, including a transposed input staged against its destination. */
internal fun assertRectangularProductAliasesAgreeWithReference(blas: KoblasEngine) {
    val rng = Random(20261014)
    for (useWorkspace in booleanArrayOf(false, true)) {
        for (aliasDestination in booleanArrayOf(false, true)) {
            val a = randomMatrix(3, 2, rng)
            val b = if (aliasDestination) randomMatrix(3, 3, rng) else DenseMatrix.wrap(2, 3, a.values)
            val c = if (aliasDestination) DenseMatrix.wrap(2, 3, a.values) else randomMatrix(3, 3, rng)
            val expected = c.copyOf()
            ReferenceBlas.gemm(0.875, a.copyOf(), aliasDestination, b.copyOf(), false, -0.25, expected)
            val workspace = if (useWorkspace) Workspace() else null

            blas.gemm(0.875, a, aliasDestination, b, false, -0.25, c, workspace)

            assertClose(expected.values, c.values, "rectangular alias=$aliasDestination workspace=$useWorkspace")
            workspace?.let { assertEquals(if (aliasDestination) 1 else 0, it.available(a.values.size)) }
        }
    }
}

/** All affected products, with independent transpose flags, stored triangles and symmetric sides. */
internal fun assertDenseProductAliasesAgreeWithReference(blas: KoblasEngine) {
    for (transposeA in booleanArrayOf(false, true)) {
        for (transposeB in booleanArrayOf(false, true)) {
            assertProductAliasesAgreeWithReference(
                { a, b, c, w -> blas.gemm(0.875, a, transposeA, b, transposeB, -0.25, c, w) },
                { a, b, c -> ReferenceBlas.gemm(0.875, a, transposeA, b, transposeB, -0.25, c) },
            )
            for (lower in booleanArrayOf(false, true)) {
                assertProductAliasesAgreeWithReference(
                    { a, b, c, w -> blas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, c, lower, w) },
                    { a, b, c -> ReferenceBlas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, c, lower) },
                )
            }
        }
        for (lower in booleanArrayOf(false, true)) {
            assertProductAliasesAgreeWithReference(
                { a, b, c, w -> blas.syr2k(0.875, a, b, transposeA, -0.25, c, lower, w) },
                { a, b, c -> ReferenceBlas.syr2k(0.875, a, b, transposeA, -0.25, c, lower) },
            )
            // The same flag also covers both positions of the symmetric operand.
            assertProductAliasesAgreeWithReference(
                { a, b, c, w -> blas.symm(0.875, a, b, -0.25, c, lower, transposeA, w) },
                { a, b, c -> ReferenceBlas.symm(0.875, a, b, -0.25, c, lower, transposeA) },
            )
        }
    }
}
