package com.eignex.koblas.sparse

/**
 * Both sparse matrix seams at once, the counterpart of [com.eignex.koblas.dense.LinearAlgebra].
 *
 * Implement this when a backend provides both, which is the usual case for a host library — SuiteSparse
 * ships the solver and the products together, unlike the CBLAS-without-LAPACKE split that made the dense
 * halves worth ranking separately. The halves are still registered independently, because costing nothing
 * to allow is better than discovering later that something needed it.
 */
interface SparseLinearAlgebra :
    SparseBlas,
    SparseLapack
