package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.F64PivotedSymmetricIndefiniteDecomposition
import com.eignex.koblas.dense.pivotedSymmetricIndefinite
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.quasiDefiniteLdl
import kotlin.test.Test
import kotlin.test.assertFalse

class LdlNamingTest {
    @Test
    fun `dense and sparse LDL expose distinct numerical contracts`() {
        val dense = F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, -2.0)))
        val sparse = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(1 to -2.0)))

        val pivoted: F64PivotedSymmetricIndefiniteDecomposition = dense.pivotedSymmetricIndefinite()
        val quasiDefinite: F64QuasiDefiniteLdlFactorization = sparse.quasiDefiniteLdl()

        assertFalse(pivoted.singular)
        assertFalse(quasiDefinite.singular)
    }
}
