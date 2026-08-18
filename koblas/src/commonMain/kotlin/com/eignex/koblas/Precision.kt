package com.eignex.koblas

/*
 * The element type is part of every container and every backend half, so it is part of their names: an
 * `F64` prefix for the double-precision ones here, leaving `F32` and `BF16` free for the siblings that
 * follow. The unprefixed names stay as aliases, so double precision is what a caller who does not say
 * which one they mean gets.
 *
 * The serial names are not prefixed. They are a wire contract, and a snapshot written before the rename
 * still reads back; a future element type gets a serial name of its own rather than sharing this one.
 */

/** Double-precision [F64VectorLike], the vector contract an unqualified `VectorLike` means. */
public typealias VectorLike = F64VectorLike

/** Double-precision [F64VectorView], the sealed vector storage an unqualified `VectorView` means. */
public typealias VectorView = F64VectorView

/** Double-precision [F64DenseVector], the dense vector an unqualified `DenseVector` means. */
public typealias DenseVector = F64DenseVector

/** Double-precision [F64SparseVector], the sparse vector an unqualified `SparseVector` means. */
public typealias SparseVector = F64SparseVector

/** Double-precision [F64MatrixLike], the matrix contract an unqualified `MatrixLike` means. */
public typealias MatrixLike = F64MatrixLike

/** Double-precision [F64MatrixView], the sealed matrix storage an unqualified `MatrixView` means. */
public typealias MatrixView = F64MatrixView

/** Double-precision [F64DenseMatrix], the dense matrix an unqualified `DenseMatrix` means. */
public typealias DenseMatrix = F64DenseMatrix

/** Double-precision [F64SparseMatrix], the CSC matrix an unqualified `SparseMatrix` means. */
public typealias SparseMatrix = F64SparseMatrix

/** Double-precision [F64Givens], the rotation an unqualified `Givens` means. */
public typealias Givens = F64Givens

/** The double-precision [F64Context], the context an unqualified `KoblasContext` means. */
public typealias KoblasContext = F64Context
