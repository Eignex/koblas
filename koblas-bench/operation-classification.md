# Numerical operation classification

Comparator status answers whether an independently bound library performs equivalent work; it does not say
whether an operation belongs to a BLAS standard. `comparator-coverage.tsv` therefore keeps direct,
composition, missing, and no-vendor-counterpart status separate from this standards classification.

| Family | Classification | Notes |
|---|---|---|
| Dense `gemmt` | standard matrix-property variant | Netlib publishes `GEMMTR`; OpenBLAS and oneMKL expose `cblas_dgemmt`. |
| Sparse `symv`, `symm`, triangular products and solves | standard matrix-property variant | These match the BLAS Technical Forum sparse matrix-property model. |
| Sparse general products | standard | Sparse-dense and sparse-sparse products retain the established `gemm` name. |
| Sparse `syrk` | vendor extension | A common rank-k product with explicit sparse or dense result storage. |
| Sparse `addScaled` | vendor extension | Sparse algebra rather than a standard BLAS root. |
| Indexed sparse level one | legacy indexed extension | `axpyi`, gather, scatter, and their benchmark counterparts. |
| Dense `sum` | vendor extension | OpenBLAS `cblas_dsum`; absent from standard BLAS and oneMKL CBLAS. |
| Packed tile products and `gemmTrsm` | implementation-private fusion | Padded microkernel panels are not conventional BLAS packed storage. |

Availability is recorded per platform by report metadata. A missing library or hardware target is unmeasured,
not evidence of parity or failure. Fresh allocation, conversion, preparation, fixture reset, and reusable-handle
execution are reported as distinct costs where the operation requires them.
