package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64MatrixLike
import com.eignex.koblas.core.F64MatrixStorage
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.core.F64VectorStorage
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceBackend
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseLinearAlgebra
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLinearAlgebra
import kotlin.reflect.KType
import kotlin.reflect.typeOf
import kotlin.test.Test
import kotlin.test.assertEquals

class PrecisionTest {

    @Test
    fun `every unprefixed alias names its F64 type`() {
        val aliases = listOf(
            alias<VectorLike, F64VectorLike>("VectorLike"),
            alias<VectorStorage, F64VectorStorage>("VectorStorage"),
            alias<DenseVector, F64DenseVector>("DenseVector"),
            alias<SparseVector, F64SparseVector>("SparseVector"),
            alias<MatrixLike, F64MatrixLike>("MatrixLike"),
            alias<MatrixStorage, F64MatrixStorage>("MatrixStorage"),
            alias<DenseMatrix, F64DenseMatrix>("DenseMatrix"),
            alias<SparseMatrix, F64SparseMatrix>("SparseMatrix"),
            alias<Givens, F64Givens>("Givens"),
            alias<ModifiedGivens, F64ModifiedGivens>("ModifiedGivens"),
            alias<KoblasContext, F64Context>("KoblasContext"),
            alias<Kernels, F64Kernels>("Kernels"),
            alias<Blas, F64Blas>("Blas"),
            alias<Lapack, F64Decompositions>("Lapack"),
            alias<LinearAlgebra, F64LinearAlgebra>("LinearAlgebra"),
            alias<ReferenceBackend, F64ReferenceBackend>("ReferenceBackend"),
            alias<SparseKernels, F64SparseKernels>("SparseKernels"),
            alias<SparseBlas, F64SparseBlas>("SparseBlas"),
            alias<SparseLapack, F64SparseDecompositions>("SparseLapack"),
            alias<SparseLinearAlgebra, F64SparseLinearAlgebra>("SparseLinearAlgebra"),
        )

        for ((name, actual, expected) in aliases) {
            assertEquals(expected, actual, name)
        }
    }

    private inline fun <reified A, reified F64> alias(name: String): Triple<String, KType, KType> =
        Triple(name, typeOf<A>(), typeOf<F64>())
}
