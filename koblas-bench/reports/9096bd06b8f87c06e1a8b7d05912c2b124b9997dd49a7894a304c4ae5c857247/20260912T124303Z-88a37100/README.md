# scal and spgather versus oneMKL and OpenBLAS

The retained fix is sparse gather dispatch. Native gather improves by a median paired **3.71x** at 1,024
stored values and **2.68x** at 16,384 stored values. The scaling investigation did not establish a reliable
production improvement; its kernel is unchanged. OpenBLAS gather is unsupported by the reference runner.

## Gather cause and fix

`NativeCIndexedSparseKernels` and `SimdIndexedSparseKernels` delegate unspecified methods to
`ScalarIndexedSparseKernels`. The three-argument `gather` interface default was one of those delegated methods.
Its receiver became the scalar delegate, so its call to the six-argument slice method bypassed the backend
implementation. The `native/c` or `simd-sparse` engine label did not mean this operation reached its C or
Vector API leaf. This also explains why experiments rewriting the C gather loop did not change its benchmark.

Moving the convenience call to an extension preserves the selected receiver. A regression test uses a
delegated implementation with an observable slice override. Native gathers below the existing 32-entry
indexed crossover use the scalar oracle to avoid pinning overhead. The original C leaf is retained.

On this AVX2 JVM the actual Vector API gather was slower than the scalar indexed loop. Gather now selects the
scalar loop explicitly for 256-bit x86 species; other indexed operations retain their own policies. The
eligibility decision is cached once. The allocation check calls the Vector API gather leaf directly so it
still tests that implementation even when production chooses scalar loads.

The final JVM gather capture is [jvm-final/jvm-Simd.csv](jvm-final/jvm-Simd.csv): arithmetic medians are 19.7 ns
for 40 entries, 257.5 ns for 1,024 entries, 260.8 ns for 655 entries, and 6,612.1 ns for 16,384 entries.
These were collected separately from the native/vendor pairs and should not be treated as simultaneous samples.

## Primary vendor comparison

All values below are medians in ns/op, pooling five samples from each of three passes. The native scalar arm
uses `KOBLAS_SPARSE_INDEXED_NATIVE_CROSSOVER=2147483647` in the same binary to reproduce the old gather route.
Scaling is identical in the two native arms and serves as a control. Each pass changes execution order.

| Arithmetic case | Native forced scalar | Native C | oneMKL | OpenBLAS |
|---|---:|---:|---:|---:|
| scal, n=4096 | 607.8 | 602.1 | 261.2 | 577.3 |
| spgather, n=4096, 1024 stored | 1246.0 | 333.8 | 295.7 | unsupported |
| spgather, n=65536, 16384 stored | 19820.0 | 7500.5 | 7206.5 | unsupported |

Computing the speedup within each pass, then taking the median, gives 3.71x (range 2.55–3.82x) and 2.68x
(range 2.63–3.97x) for gather. The median paired native/oneMKL time ratios are 1.15 and 1.02 respectively.
The 1,024-entry native/oneMKL ratio varies from 1.06 to 2.30; the larger case varies from 1.01 to 1.05.
Keep that variation visible rather than claiming universal parity. See [tables.md](tables.md) and the
[paired CPU trace](paired/cpu.csv).

## Scaling investigation

The initial, less-loaded capture had n=4096 arithmetic medians of 262 ns for JVM SIMD, 245 ns for Native,
278 ns for OpenBLAS, and 357 ns for oneMKL. The later paired native control above trails oneMKL by 2.3x and
roughly matches OpenBLAS. Other captures changed those rankings again. This does not establish that scaling
is universally competitive or universally slow.

Three candidates were investigated and excluded:

- Four-vector JVM loop unrolling did not establish a repeatable improvement and hurt short cases.
- Routing long JVM scales through HotSpot's autovectorized scalar loop did not establish a dependable
  improvement against the vendors under changing load.
- A controlled C offset probe identified cache-line-split stores as a possible native bottleneck. Peeling
  a prefix to align AVX2 stores helped that diagnostic, but the actual native benchmark did not confirm
  the gain. In three alternating passes its median before/after speedups were below one at all five
  tested scaling cases. The implementation and its trial-only tests were removed.

The last experiment's raw before/after/vendor samples are in [alignment-trial](alignment-trial/). It uses
separate old and candidate native binaries and the same benchmark cases. A quiet rerun with stable core
frequency is needed before choosing a scaling implementation change. The benchmark cases and matching
vendor support are retained to make that follow-up reproducible.

## Timing boundaries

Default `scal` resets its vector and multiplies by 0.875. `scal+...+timing=arithmetic` multiplies by -1,
which preserves fixture magnitudes over arbitrarily many calls. These are different coefficient workloads;
do not subtract their timings to claim an exact reset cost.

Default `spgather` now resets only the dense source, matching oneMKL. Its arithmetic variant resets neither
buffer because gather overwrites the sparse values and leaves its source alone. Both runners consume the
first and last output values. The final Kotlin consumer uses primitive indexed reads to avoid nullable
boxing on Native. Historical gather results before commit 88a37100 included an extra Koblas output reset.
Compare matching timing boundaries and account for that historical difference.

The old 2.9 microsecond native result at n=4096 and density=0.01 did not reproduce in the initial capture.
The confirmed dispatch defect is most visible with larger stored counts.

## Captures and provenance

[hardware.txt](hardware.txt) identifies the i9-12900H host. Every runner is pinned to logical CPU 4, a P-core,
and vendors use one thread. JVM runs use JDK 25.0.1 and JMH 1.37; Native uses Kotlin 2.4.10. Vendor run records
identify OpenBLAS and oneMKL versions. No raw sample values have been edited or discarded.

| Directory | Source | Purpose |
|---|---|---|
| baseline | 88a37100 | Matching arithmetic cases with original kernels and original gather dispatch |
| unrolling-trial | 909088ed | Rejected explicit scaling and C gather loop experiments |
| dispatch-trial | 89e48bc3 | Fixed gather dispatch, actual JVM Vector API gather, rejected scaling crossover |
| loaded-host | 5e5a7f1f | Gather fix and explicit AVX2 scalar selection; scaling restored |
| paired | 6e4c009a | Alternating forced-scalar/native/vendor comparison |
| jvm-final | 6e4c009a | Final JVM gather with cached eligibility |
| alignment-trial | 5e5a7f1f / a5d0ea79 | Original and rejected aligned native scaling binaries with vendors |

The normal captures use five warmups, five samples, a 200 ms target, and two JVM forks. Native and vendor
warmups target 50 ms; JMH warmups target 200 ms. Paired captures use three warmups, five samples, a 50 ms
measurement target, and three passes; native/vendor warmups target 12.5 ms. Native calibration can exceed
the target and vendor calibration has its own operation cap. CSV records retain those boundaries.

CPU traces show large changes in background load, exceeding 70% system utilization in the loaded-host
capture. Initial timings and later absolute timings are not a stable before/after experiment. Use the
short alternating comparisons for the gather conclusion. `times.txt` files record execution starts; the
saved capture scripts define their order. The final JVM-only rerun has timestamps but no separate CPU trace.
`dirty=true` in the alignment trial reflects untracked report artifacts; the kernel changes were committed.

## Reproduce and verify

[reproduce.sh](reproduce.sh) runs this report's [cases.txt](cases.txt) through the four Koblas arms and both
vendors into a new output directory. Run it from the repository root. The paired scripts retain the exact
commands used, including the native scalar override and old-binary comparison; their temporary paths need
adjusting to repeat a historical capture.

Verification passed with:

```bash
./gradlew :koblas:check :koblas-bench:check lintDocs check
./gradlew :koblas:jvmTest :koblas-bench:jvmTest -Pkoblas.noSimd=true
JAVA_TOOL_OPTIONS=-XX:MaxVectorSize=16 ./gradlew :koblas:jvmTest \
  --tests 'com.eignex.koblas.sparse.IndexedSparseKernelsTest' \
  --tests 'com.eignex.koblas.sparse.JvmVectorScatterTest'
```

The final check is repeated after restoring the original scaling kernel. Public API snapshots are unchanged.
