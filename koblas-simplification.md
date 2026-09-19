# Koblas portable restoration

Updated 2026-09-19. This repository record supersedes the former vendor-only simplification plan. The
authoritative architecture and staged sequence are the user-owned `koblas-plan.md`, `koblas-sme.md`, and
`koblas-sme-steps.md` in the workspace; their historical SME filenames no longer describe their goal.

Koblas provides portable Kotlin dense and sparse BLAS Levels 1–3 without requiring an installed numerical
library. Owned JVM performance uses Kotlin and the Vector API behind generic logical panel and tile contracts.
Kotlin/Native keeps the portable correctness floor and may use optional installed host bindings where a real
binding supports the operation. Every invocation uses one compute thread, and independent calls own mutable
scratch.

Custom numerical C/C++/assembly, SME/SME2 code, native probe/catalog/build machinery, speculative ISA
infrastructure, bundled vendor payloads, process-global provider installation, factorization policy, HFactor,
and solver workflows stay removed. Thin bindings to installed libraries remain explicit alternatives and retain
their resolved binary path, identity, available version/threading evidence, and concrete call route.

## Stages

- S1 restores validated portable dense GEMV, SYMV, rank updates, triangular operations, GEMM, GEMMT, SYMM, SYRK,
  and SYR2K; defines generic matrix-product APIs; makes exact scalar/SIMD/host benchmark attribution truthful.
- S2 restores sparse Level 2/3, prepared snapshots, and all dense/sparse product pairings without densification.
- S3 adds architecture-selected logical panel kernels and owned JVM Level 2 acceleration.
- S4 adds JVM GEMM tiles, layouts and cache-blocked products.
- S5 adds structured product and triangular JVM execution.
- S6 accelerates sparse dense-RHS panels and prepared reuse.
- S7 integrates optional Native host acceleration without making it an availability dependency.
- S8 calibrates defaults across real hardware and audits final coverage and attribution.

Later stages are not implied complete by S1. In particular, an exact JVM SIMD engine may truthfully report a
portable scalar Level 2/3 component until the corresponding owned kernel stage lands. A requested arm name is
never evidence of execution, and `koblas.noSimd` only withholds the Vector API module; vendor absence is verified
separately.

Verification for every implementation stage is `./gradlew check lintDocs` plus the same command with
`-Pkoblas.noSimd=true`, genuine vendor-absent portable execution, runnable configured Native tests, and explicit
reporting of cross-compiled but unexecuted targets. External host conformance remains optional and cannot gate
mandatory portable coverage.
