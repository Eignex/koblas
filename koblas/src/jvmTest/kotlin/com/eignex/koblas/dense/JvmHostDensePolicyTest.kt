package com.eignex.koblas.dense

import com.eignex.koblas.internal.jvm25.JvmHostDensePolicy
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class JvmHostDensePolicyTest {
    @Test
    fun `all product extents must amortize operand transfers`() {
        for (operation in operations) {
            assertTrue(JvmHostDensePolicy.eligible(operation, 128, 128, 128), "$operation")
            for ((rows, columns, depth) in listOf(
                Triple(127, 512, 512),
                Triple(512, 127, 512),
                Triple(512, 512, 127),
            )) {
                assertFalse(
                    JvmHostDensePolicy.eligible(operation, rows, columns, depth),
                    "$operation $rows $columns $depth",
                )
            }
        }
    }

    @Test
    fun `level two windows and retained panels keep owned arithmetic`() {
        for (operation in DenseMatrixOperation.entries.filterNot { it in operations }) {
            assertFalse(JvmHostDensePolicy.eligible(operation, 512, 512, 512), "$operation")
        }
    }

    @Test
    fun `deep gemm stays within the measured whole matrix extents`() {
        for (depth in listOf(1024, 4096, 16_384)) {
            assertTrue(JvmHostDensePolicy.eligible(DenseMatrixOperation.Gemm, 16, 16, depth))
        }
        for ((rows, columns, depth) in listOf(
            Triple(15, 64, 4096),
            Triple(64, 15, 4096),
            Triple(16, 16, 1023),
            Triple(16, 16, 16_385),
        )) {
            assertFalse(JvmHostDensePolicy.eligible(DenseMatrixOperation.Gemm, rows, columns, depth))
        }
    }

    @Test
    fun `borrowed deep products keep the larger cutoff for extra packing`() {
        assertFalse(JvmHostDensePolicy.eligibleBorrowed(DenseMatrixOperation.Gemm, 16, 16, 4096))
        assertTrue(JvmHostDensePolicy.eligibleBorrowed(DenseMatrixOperation.Gemm, 128, 128, 4096))
    }

    private val operations = listOf(
        DenseMatrixOperation.Gemm,
        DenseMatrixOperation.Gemmt,
        DenseMatrixOperation.Symm,
        DenseMatrixOperation.Syrk,
        DenseMatrixOperation.Syr2k,
        DenseMatrixOperation.Trmm,
        DenseMatrixOperation.Trsm,
    )
}
