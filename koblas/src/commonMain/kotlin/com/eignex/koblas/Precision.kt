package com.eignex.koblas

import com.eignex.koblas.core.*

/*
 * The element type is part of every container and backend half, so it is part of their expert-facing names.
 * The unprefixed aliases make double precision the convenient default without hiding the concrete F64 API.
 */

/** Double-precision [F64VectorLike], the vector contract an unqualified `VectorLike` means. */
public typealias VectorLike = F64VectorLike

/** Double-precision [F64VectorStorage], the sealed storage an unqualified `VectorStorage` means. */
public typealias VectorStorage = F64VectorStorage

/** Double-precision [F64DenseVector], the dense vector an unqualified `DenseVector` means. */
public typealias DenseVector = F64DenseVector

/** Double-precision [F64SparseVector], the sparse vector an unqualified `SparseVector` means. */
public typealias SparseVector = F64SparseVector

/** Double-precision [F64MatrixLike], the matrix contract an unqualified `MatrixLike` means. */
public typealias MatrixLike = F64MatrixLike

/** Double-precision [F64MatrixStorage], the sealed storage an unqualified `MatrixStorage` means. */
public typealias MatrixStorage = F64MatrixStorage

/** Double-precision [F64DenseMatrix], the dense matrix an unqualified `DenseMatrix` means. */
public typealias DenseMatrix = F64DenseMatrix

/** Double-precision [F64SparseMatrix], the CSC matrix an unqualified `SparseMatrix` means. */
public typealias SparseMatrix = F64SparseMatrix

/** Double-precision [F64Givens], the rotation an unqualified `Givens` means. */
public typealias Givens = F64Givens

/** The double-precision [F64Context], the context an unqualified `KoblasContext` means. */
public typealias KoblasContext = F64Context
