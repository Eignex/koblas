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
