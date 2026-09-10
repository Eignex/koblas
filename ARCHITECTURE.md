# Koblas architecture

Koblas has one immutable engine composition. The platform default is created once as `koblas`; explicit scalar,
C, and SIMD contexts are created through `BuiltinKernels` without changing that default. Every matrix algorithm
receives its selected dependencies when its context is constructed, including packed-panel helpers.

```text
public operations and conveniences
    -> validation, aliases, scratch lifetime, shapes and traversal
        -> DenseVectorKernels              contiguous standalone Level 1
        -> IndexedSparseKernels            ordered raw sparse slices for matrix work
        -> SparseKernels                   standalone SparseVector Level 1
        -> DensePanelKernels               matrix column and panel arithmetic
        -> portable access kernels         strided/generic vectors and matrices
        -> packed layout + PackedKernels   padded panels and fixed-shape tile arithmetic
        -> sparse panel kernels            CSC column and dense-RHS arithmetic
        -> sparse accumulation kernels     support-aware product, merge and workspace arithmetic
```

The public `PackedPanels` helper is an engine-bound value available as `KoblasContext.packedPanels`; the
top-level `packedPanels` value is only the convenience bound to the default `koblas` context. Explicit-engine
code must use the context value so tile shape, packing, and tile arithmetic stay in one composition.

## Loop ownership

| Responsibility | Owner |
| --- | --- |
| Dense and generic Level 2/3 arithmetic | dense access, GEMV, symmetric, ordered-product, rank-update and triangular kernel files |
| Dense storage layout, row gathering, scaling and matrix reductions | `DenseStorageKernels.kt` |
| Generic and strided Level 1 arithmetic | `VectorAccessKernels.kt` |
| Packed padding, packing and writeback | `PackedLayout.kt`, behind checked engine-bound panel operations |
| Packed GEMM/TRSM arithmetic | the selected `PackedKernels` implementation |
| Sparse matrix column/RHS arithmetic | `IndexedSparseKernels` and `SparsePanelKernels` |
| Sparse product, merge, rank-update and workspace values | `SparseAccumulationKernels.kt` |
| Sparse storage layout, row gathering, scaling and matrix reductions | `SparseStorageKernels.kt` |

Scalar loops remain inside portable, C, and SIMD kernel implementations. Algorithm files may retain block and
solve scheduling, shape decisions, CSC column walks, support sorting, capacity growth, validation, and
exception-safe scratch ownership. Those loops determine structure or lifetime; they do not duplicate numerical
leaves. Operation-specific exceptional-value policy also stays at the algorithm boundary so GEMMT, SYRK,
SYR2K, SYMV, and triangular operations preserve their distinct evaluation orders.

Common algorithms contain no FFM, Vector API, or native imports. JVM and Native source sets provide the static
platform families. The optional `koblas-hfactor` module depends on `koblas`; the core module never discovers or
loads HFactor.

## Benchmark boundary

`koblas-bench` is not a production dependency. Its flat Kotlin runner measures exactly selected `jvm-c`,
`jvm-simd`, or `native` contexts through Gradle. Standalone Bash and C runners measure OpenBLAS and oneMKL into
separate CSV files. Both sides consume `cases.txt`; hardware collection and CSV comparison remain standalone
shell tools. Packed cases record the physical shape and use the selected context's bound panel operations.
Allocation contracts are verified only by unit tests.
