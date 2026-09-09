# Saved benchmark reports

These archives preserve benchmark evidence that would otherwise be deleted with the Gradle build directory.
They may contain machine and checkout details; inspect their metadata before sharing them outside the project.

## Retained BLAS gaps

`blas-gaps-20260909.tar.gz` preserves two original JVM passes for packed `gemmt` against independently bound,
single-threaded OpenBLAS and two original passes over all nine retained sparse completion rows. It also contains
two clean-commit follow-up passes for the corrected GEMMT eligibility scan and sparse SYRK adjacency traversal.
The runs used an Intel Core i9-12900H with affinity `2-5` on a shared host. JMH's advisory global lock was ignored
where another benchmark was active; raw confidence intervals and cross-pass variation are retained.

OpenBLAS 0.3.30 was available and resolved directly. oneMKL was unavailable, so no replacement arm is reported.
The built-in/OpenBLAS `gemmt` intervals overlap at the measured `129x257` lower-triangle case. The corrected
adjacency traversal removes the old sparse SYRK structural-search cost. Detailed source provenance, exact commands,
results, and validation status are in `blas-gaps-20260909-notes.md` and inside the archive.

Expected SHA-256:
`f80907b5690f6cc2767932155d0410c12edcae2cb3b00f96cd858bbc5264eaa2`.

## Standard hardware profile

`hardware-standard-18286e42-jvm-20260909.tar.gz` is the first complete version-1 contributor-profile report.
It was produced from clean commit `18286e42bc6d9b39c0ca2d8513ce2433dc026ab4` on an Intel Core i9-12900H
with affinity `0,2,4,6`. Two fresh built-in passes cover all 83 bounded dense and sparse cases, and two matched
single-threaded OpenBLAS 0.3.30 passes cover the 54 cases with that comparator. The runner recorded 245.1 seconds
elapsed on a shared host and did not detect another contributor-runner invocation; unrelated machine activity was
not excluded. The archive includes the workload catalog, raw JSON and logs, stable aggregate rows, metadata,
coverage manifests, a readable summary, and per-file checksums.

Validate it offline with:

```bash
koblas-bench/report.sh validate koblas-bench/results/hardware-standard-18286e42-jvm-20260909.tar.gz
koblas-bench/report.sh summarize koblas-bench/results/hardware-standard-18286e42-jvm-20260909.tar.gz
```

Expected SHA-256:
`4e6f7108c0da746546aee883e56dbb4e4f70e7b0647b9c8ede1709f4eb761f48`.

## Packed syr2k

`syr2k-a7272cee-jvm-20260908.tar.gz` contains the raw JMH JSON used to compare the packed built-in `syr2k`
implementation at commit `a7272ceea6eb11ff39bc07a55ea94cc945a9ac73` with single-threaded OpenBLAS and
oneMKL 2026.1. It includes:

- two independent built-in passes;
- two independent OpenBLAS passes;
- one stable oneMKL pass and two retained noisy oneMKL passes;
- benchmark metadata and the derived timing ratios.

The run used an Intel Core i9-12900H with CPU affinity `0,2,4,6`. The benchmark covered `9x5`, `128x129`,
`257x31`, and `512x127` inputs across both triangle and transpose variants, with three 500 ms warmups and five
500 ms measurements per case.

Verify and inspect the archive with:

```bash
sha256sum koblas-bench/results/syr2k-a7272cee-jvm-20260908.tar.gz
tar -xzf koblas-bench/results/syr2k-a7272cee-jvm-20260908.tar.gz
```

Expected SHA-256:
`b1d3a26a4f5452317cfc2e72d8fff32ecf3aa2616dd05ba17bff6b85da3d4a17`.

## Packed panels

`packed-panels-6fff651a-jvm-20260909.tar.gz` preserves two independent JMH passes for the public packed-panel
pack and write helpers. It also includes an untouched-main reproduction of the small-depth `gemmTile` allocation
bug, two successful post-fix passes, and two successful passes after moving the SIMD tile into its own file at
commit `6fff651a`. Complete console logs, the temporary baseline harness, metadata, and paired timing results are
included.

The run used an Intel Core i9-12900H with CPU affinity `0,2,4,6` on a busy shared host. It covered depths
`3`, `31`, and `128` at full `8x4` and partial `7x3` tile edges. Every pack and write allocation probe reported
`0 B/call`; confidence intervals and cross-pass variation are retained in the archive. OpenBLAS and oneMKL do
not accept koblas microkernel-packed panels, so there is no meaningful external comparator for these helper-only
operations.

Verify and inspect the archive with:

```bash
sha256sum koblas-bench/results/packed-panels-6fff651a-jvm-20260909.tar.gz
tar -xzf koblas-bench/results/packed-panels-6fff651a-jvm-20260909.tar.gz
```

Expected SHA-256: `737baf460d7c7f06d3fe70f3c799f4f4b06aac279ed0e93dd0aa86517efbaa30`.

## Fused level-2 kernels

`fused-level2-20260909.tar.gz` contains the raw JMH JSON for the fused `axpy4` and `dotAxpy` kernels
and their GEMV/SYMV callers in this change. It includes:

- two JVM Vector API microbenchmark passes with the unfused compositions;
- two square GEMV/SYMV passes against single-threaded OpenBLAS and oneMKL;
- one tall, wide, and remainder-heavy GEMV pass against both comparators; and
- one Kotlin/Native pass comparing the compiled-in vectorized C kernels with the scalar reference.

The runs used an Intel Core i9-12900H with CPU affinity `0,2,4,6`. Other builds and benchmarks remained
active on the shared machine, so the error bars are part of the result rather than an idle-host claim. JMH's
lock check was disabled where another benchmark process already held the global lock. OpenBLAS and oneMKL
were fixed to one thread; oneMKL used its sequential threading layer.

The JVM allocation probes measured 0 B/call for both new Vector API kernels. Across the two square passes,
the built-in 1024-order SYMV took 151--155 us for the lower triangle, against 98--99 us for OpenBLAS and
83--108 us for oneMKL. This leaves a visible comparator gap, but replaces the roughly 571 us prior built-in
measurement. Rectangular GEMV was at or near comparator time within the observed noise. On Kotlin/Native,
the vectorized C `axpy4` took 163 ns at length 256 and 447 ns at 1024, while four C AXPYs took 342 ns and
1094 ns; `dotAxpy` took 113 ns and 278 ns, versus 161 ns and 403 ns for separate C dot and AXPY calls.

Verify and inspect the archive with:

```bash
sha256sum koblas-bench/results/fused-level2-20260909.tar.gz
tar -xzf koblas-bench/results/fused-level2-20260909.tar.gz
```

Expected SHA-256:
`3c975c291509dfc9dce8e4e7846a75864d10d560b6a227ed0f0a789d65cbe950`.

## Packed triangular solve

`packed-trsm-20260909.tar.gz` retains raw JVM and Kotlin/Native JSON for packed TRSM and fused GEMM-TRSM,
forced packed/scalar crossover runs, and single-threaded OpenBLAS and oneMKL comparisons. Its notes record
allocation probes, implementation identities, the measured thresholds, and variability from the busy shared
host.

Expected SHA-256:
`00978425a758cf1608727482f3401fa64dfe8279d024d30533ce5db27e21249c`.

## Packed triangular SIMD edges

`packed-triangular-simd-0320cab7-jvm-20260909.tar.gz` retains the fresh before/after JVM Vector API
measurements for logical-edge packed GEMM-TRSM tiles at depths 3, 31 and 128. It contains two corrected
benchmark-only baseline passes, two candidate passes, an end-to-end built-in/OpenBLAS TRSM pass, and an
unchanged Linux x86-64 scalar/C validation run using the corrected platform-shaped fixture.

The run used an Intel Core i9-12900H with affinity `0,2,4,6` while another JMH process was active. Raw
confidence intervals and cross-pass variation are retained. The lower non-unit fixture stores diagonal
1.25 and performs division; its depth-3 result is inconclusive, while depth 31 improves 2.81x to 3.31x and
depth 128 improves 8.76x to 8.80x across the two passes. Upper unit-diagonal edges improve at every depth.
Every successful warmed allocation probe reports `0 B/call`. OpenBLAS was single-threaded. oneMKL, ARM and
macOS were unmeasured, and the end-to-end results retain substantial residual gaps rather than claiming parity.

Expected SHA-256:
`50d532c7f6f243b9af7d664cbb964472cd0cb84f85dc954b020cf23ddc4c1df9`.

## Sparse workspace and C dispatch

`sparse-workspace-dispatch-20260909.tar.gz` preserves two independent JVM sparse-workspace passes, two Native
passes, two JVM-without-SIMD level-1 C crossover passes, raw and installed-policy packed tile measurements, two
eligibility scans, and two repeated end-to-end JVM packed-TRSM passes. The archive includes raw JSON, complete
console logs, resolved implementation identities, allocation probes, and notes describing the scenario-specific
fallback outcomes.

The runs used an Intel Core i9-12900H with CPU affinity `0,2,4,6` on a busy shared host. JVM allocation probes
reported `0 B/call`. The retained evidence supports JVM C crossovers of 64 for `axpy4`, 256 for `dotAxpy`, and
depth 16 for packed product and fused update/solve tiles. Standalone packed TRSM remains portable because its C
leaf did not win consistently. The #494 eligibility predicate is unchanged: constructed dense-finite inputs stay
eligible, while constructed structural-zero and overflow-bound inputs exercise the fallback.

Expected SHA-256:
`969aef36f3a630d2553616d7d1b85f557af36421bb32e3d242eafbcfcefa189a`.

## Sparse workspace equivalent comparisons

`sparse-workspace-comparators-20260909.tar.gz` retains two affinity-pinned JVM passes and one Linux x86-64 Native
pass for the contract-equivalent sparse workspace comparison suites. The JVM passes cover full built-in operations,
separately reported destructive fixture resets and primitive-only partial work, plus independent Kotlin baselines;
every warmed JVM allocation probe reports `0 B/call`. The Native pass covers the no-vendor-counterpart baselines and
growing-support workload, with managed allocation explicitly unavailable.

The host was shared and concurrent work was allowed. Cross-pass JVM variation is visible in the raw JSON and is not
collapsed into a parity claim. `libmkl_rt` was unavailable, so the archive contains no oneMKL measurements and no
substitute comparator. The included notes record commands, workload dimensions, seed, identities, coverage gaps,
and concise findings. Contributor profile v1 remains unchanged; these are selected developer results.

Expected SHA-256:
`56b342dc614d837fab64a09388c739e002446b24c94c8d85ee54d8dfc92fa2fa`.

## Packed triangular multiply

`packed-trmm-20260909.tar.gz` retains two independent JVM SIMD/OpenBLAS passes across five shapes and all
six storage, transpose, unit-diagonal and side variants; forced packed/reference crossover sweeps; JVM
without SIMD; and a Kotlin/Native C/OpenBLAS pass. The raw files and detailed notes preserve confidence
intervals, resolved implementations, the busy-host variance, and the unavailable oneMKL comparator.

The evidence supports order 16 and panel width 32 as conservative packed-dispatch boundaries. The 64x32
region remains at parity within uncertainty, while the 128x256 packed path materially improves on the
reference traversal. OpenBLAS remains faster on both JVM and Native, and the archive records that residual
gap rather than presenting comparator parity.

Expected SHA-256:
`382da237112a3645a25489e7525f47ef116aac0bc28ff17abb559a64d5a0fc0c`.
