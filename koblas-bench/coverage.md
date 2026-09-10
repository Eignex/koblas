# Benchmark coverage and external mapping

The shared workload keeps representative public BLAS, sparse, workspace, panel, and packed operations while
removing parameter-class duplication. `direct` means one documented vendor call performs the arithmetic;
`composed` means documented calls reproduce the retained mathematical work; `partial` means only a useful
arithmetic subset is comparable; `unsupported` retains the koblas case without a defensible vendor equivalent.

| cases | OpenBLAS | oneMKL | external timing boundary |
| --- | --- | --- | --- |
| dot, axpy, axpy-arithmetic, scal, nrm2, asum, iamax, swap, rot, rotm, rotmg | direct | direct | CBLAS call; destructive input reset included on both sides |
| sum | direct OpenBLAS extension | unsupported | arithmetic call |
| compensated-sum, ssqd | unsupported | unsupported | koblas-only compensated/fused reductions |
| dot4 | composed as four DDOT | composed as four DDOT | four calls |
| axpy4 | composed as four DAXPY | composed as four DAXPY | reset plus four calls |
| dot-axpy | composed DDOT+DAXPY | composed DDOT+DAXPY | reset plus both calls |
| gemv, symv, ger, syr, syr2, trsv, trmv | direct | direct | reset where destructive plus one CBLAS call |
| gemm, symm, gemmt, syrk, syr2k, trsm, trmm | direct | direct | reset plus one CBLAS call |
| gemm-tile full and logical edge, physical 4x4 and 8x4 | partial DGEMM arithmetic | partial DGEMM arithmetic | packed values are staged as the same logical column-major operands before timing; arithmetic only |
| packed-trsm full and logical edge, physical 4x4 and 8x4 | partial right-side DTRSM arithmetic | partial right-side DTRSM arithmetic | packed triangle/RHS staged before timing; arithmetic only |
| gemm-trsm full and logical edge, physical 4x4 and 8x4 | composed DGEMM then DTRSM | composed DGEMM then DTRSM | staging excluded; update and solve both timed; signs/scaling match `B - A*R`, then `X*T=B` |
| pack-left/right full and edge | unsupported | unsupported | koblas layout only; vendor packing would create an unmatched private format |
| pack-symmetric-left/right | unsupported | unsupported | koblas selected-triangle layout only |
| pack-triangular-left/right | unsupported | unsupported | koblas structure/unit-diagonal materialization only |
| write-left/right full and edge | unsupported | unsupported | koblas layout only |
| clear-left/right-padding full and edge | unsupported | unsupported | koblas physical-padding maintenance only |
| spdot, spaxpy, spscatter, spgather, spgather-zero | unsupported | direct legacy indexed BLAS | reset plus arithmetic for destructive operations |
| spdot-sparse, spnrm2, spasum | unsupported | unsupported | retained koblas sparse-vector reductions |
| spgemv prepared/oneshot | unsupported | direct inspector-executor MV | prepared conversion/handle outside timing; one-shot CSC-to-CSR conversion and create/optimize/destroy included |
| spmm prepared/oneshot | unsupported | direct inspector-executor MM | same conversion and ownership split; dense reset included |
| spgemm prepared/oneshot | unsupported | partial sparse SPMM | input conversion follows mode; fresh output is consumed only through its handle and its lifetime is timed |
| spsymv | unsupported | direct sparse MV with symmetric descriptor | one-shot handle lifetime and call |
| spsymm | unsupported | direct sparse MM with symmetric descriptor | one-shot handle lifetime, reset, and call |
| sptrsv, sptrmv | unsupported | direct sparse TRSV / MV with triangular descriptor | one-shot handle lifetime and call |
| sptrsm, sptrmm | unsupported | direct sparse TRSM / MM with triangular descriptor | one-shot handle lifetime, reset, and call |
| spsyrk-dense, spsyrk-sparse | unsupported | direct SYRKD / partial sparse SYRK | fresh sparse output is consumed only through its handle and its lifetime is timed |
| spadd | unsupported | partial sparse ADD | fresh output is consumed only through its handle and its lifetime is timed |
| workspace scatter/gather/max/filter families | unsupported | unsupported | retained caller-owned koblas workspace contract; an indexed primitive alone is not equivalent |

Packed physical cases deliberately include each supported koblas shape on this acceptance host: native/JVM C
4x4 and JVM SIMD 8x4, each with a complete logical tile and a partial edge. A mode emits unsupported for the
other physical shape. This makes different padded work impossible to join under one case ID. Panel layout cases
likewise retain full/edge packing, writeback, symmetric/triangular recipes, and padding restoration even though
no vendor layout claim is available.

## Intentional exclusions

Overload, operator, and convenience aliases are represented by the underlying operation case rather than a
second timing row. Strided and generic views reach the same portable access kernels and remain correctness-test
concerns. Row/column slicing, `withColumn`, transpose, copying, storage scaling, zeroing, and dense/sparse matrix
norms are storage transformations or simple traversals rather than backend-comparison operations; their existing
unit tests remain the useful guard. Scalar rotation construction `rotg` is represented by the `rot` application;
modified construction has its own `rotmg` case because it is dispatched. Fresh sparse SYR/SYR2 structure
construction and right-side sparse transpose compositions are excluded from this bounded workload: they have no
direct vendor equivalent, while the retained sparse product, add, rank-k, triangular, and workspace cases cover
the underlying accumulation, output construction, and transpose-sensitive arithmetic families.
