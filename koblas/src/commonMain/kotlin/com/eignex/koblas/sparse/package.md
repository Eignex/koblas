# Package com.eignex.koblas.sparse

The sparse package contains validated CSC containers and sparse BLAS levels 1–3. General products support sparse
or direct dense destinations. Symmetric products read exactly one selected CSC triangle, while triangular products
and solves use explicit lower, transpose, unit-diagonal, and side flags. Sparse `syrk` returns either one triangle
of a dense destination or a fresh selected-triangle CSC result; general products do not implicitly mirror it.

`addScaled` and sparse `+`/`-` are sparse algebra extensions. Prepared products retain immutable owned snapshots;
transpose indexing may be cached without introducing descriptors or mutable global selection.

Factorization and basis-solver contracts live in the optional JVM `koblas-hfactor` artifact. The multiplatform
BLAS module neither loads nor detects HFactor.
