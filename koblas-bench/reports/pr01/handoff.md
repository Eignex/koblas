# PR 01 baseline handoff

[PR #540](https://github.com/Eignex/koblas/pull/540) contains the final base/head, CI results and independent
review. Implementation session: `01a0910f-6e18-7353-82d3-d1de5ca821a8`, `gpt-6-astra` / `high`.
Backward API compatibility is not a requirement. The benchmark-only JVM friend adapter is owned for removal
by PR 07. See [contracts](../../contracts.md) and [packed cases](../../packed-cases.md) for the semantic
inventory and workload 5 / fixture 2 / CSV schema 4 definitions.

## Retained measurements

Hardware fingerprint: `f342c3192a41a07cf0454d7d4d5324f8289cc01329bcc9c990b986dcdd2f5922`.
Both capture directories below are beside this directory under that fingerprint:

| Capture | Source | Cases | Samples per Koblas mode | OpenBLAS samples |
| --- | --- | --- | --- | --- |
| `20260911T153950Z-a038f25ac668` | `a038f25ac66821574b30392dc1d3b189cc287e2c` | 143 | 585 across 117 supported cases | 280 across 56 supported cases |
| `20260911T155812Z-50f1356ff734` | `50f1356ff734c61f359e46e821b0f580d6549e73` | 52 packed | 130 across 26 supported cases | 80 across 16 supported cases |

Raw CSVs, exact case snapshots, hardware/toolchain provenance and completion status are retained. Both runs
completed sequentially: JVM scalar, JVM C, JVM SIMD, Native C, OpenBLAS. Settings: three warmups, five samples,
200 ms target, one JVM fork, pass 1. The host was an Intel Core i9-12900H on Linux x86-64; benchmark runtimes
were Adoptium 25.0.1, Kotlin/Native 2.4.10, Native Clang 21.1.6 and OpenBLAS 0.3.32. The shell Java version in
provenance is separate from the benchmark JVM recorded in CSV rows.

Both source patches were empty. CSV dirty flags reflect report creation; a later geometry-selector change
was test-only. The packed suite was recaptured after fixed helper arguments were pinned. Source SHAs and raw
measurements are unchanged. Main commit `2418477c` subsequently changed SIMD dot-AXPY; preserved timings for
that operation describe the recorded earlier source, not the updated main implementation.

No affinity, host exclusivity or thermal stabilization was enforced; some within-case spreads approach 2.6x.
These initial baselines do not calibrate thresholds or establish stable speedups. Hosted macOS CI executed
core JVM and Arm Native correctness tests, but no Apple benchmark or M4 measurement was captured. SME/SME2
execution remains outstanding. The oneMKL smoke only validated its runner/library.

Use [the comparator](../../README.md#save-and-compare-results) to regenerate comparisons: fixed mode for
packed scalar/C and C/Native pairs, logical mode for C/SIMD blocks, and logical `prepacked-compute` for vendor/C.

## Verification

`./gradlew check` and `./gradlew :koblas:check :koblas-bench:check lintDocs` passed, including the latter with
`-Pkoblas.noSimd=true`. Altered-default/vector-width tests, all seven comparator tests, OpenBLAS parser/full
workload checks and the eleven-case oneMKL smoke passed. Independent review (`/root/independent_review`, fresh
context, `gpt-6-astra` / `xhigh`) resolved three findings: pin helper arguments and independently verify packed
formulas; select compatible vendor timing; reject unlisted SIMD geometries. The PR record holds the current
review verdict and CI links.

Routine logs, test XML, empty patches, audit output and derived comparisons are omitted from the repository
tree. The original evidence is preserved at source commit `f3a8a5958aa20606a4fccb3a209b4e796379a023` and in the
local archive `/home/rasmus/Workspaces/koblas-pr01-evidence-f3a8a595.tar.gz` (SHA256
`5bb1d35d94ce4390667da65d71a0562ad6f6011924ecf61f281a82556988416a`). Generated report bundles are ignored by Git.
