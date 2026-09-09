# Dense Level 2 tuning notes

These measurements started from clean `origin/main` at `d39aefbea455fa900d07110a465e93902a88cd55` and then
measured the uncommitted implementation under review. They ran on an Intel Core i9-12900H with affinity pinned to
CPU 4. OpenBLAS was forced to one thread with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Other processes
remained active on the shared host; raw confidence intervals and independent-pass variation are retained rather
than treating the machine as idle.

oneMKL was unavailable because the host has no loader-visible `libmkl_rt`. ARM and macOS were not available for
measurement. Those combinations are unmeasured, and no substitute result is claimed.

## Coverage

The retained JVM files cover built-in SIMD, explicit SIMD, explicit bundled C, scalar fallback and
single-threaded OpenBLAS. They include square orders 64 through 2048, the 64/96/128/192/256 crossover sweep,
orders 512/1024/2048 on both SYMV triangles, both GEMV transpose modes, and tall, wide and remainder-heavy GEMV
shapes. The Linux x86-64 files compare the compiled-in C kernels with native OpenBLAS for both triangles at
orders 512, 1024 and 2048. Every timed case uses three 500 ms warmups and five 500 ms measurements.

Correctness was separately checked against the explicit scalar engine for lower and upper selected triangles,
zero vector entries, NaN/infinity arithmetic, poisoned unselected storage and the 512 boundary plus remainder.
Existing alpha/beta and shape suites were also run with JVM SIMD and `-Pkoblas.noSimd=true`.

## Findings

- Existing four-column GEMV is retained. In the initial rectangular JVM pass, built-in non-transposed GEMV was
  13.3 us/op against OpenBLAS 18.9 at 64x2048, 12.3 against 21.5 at 2048x64, and 79.0 against 89.4 at 2047x257.
  Transposed built-in results were also ahead in the stable rectangular cases. Large square results overlap amid
  substantial contention, so no new output-blocking claim is made.
- Transposed GEMV previously borrowed a four-double array when no workspace was supplied. Its four dot results
  now temporarily occupy the destination entries whose old values are held in scalar locals. The warmed JVM
  allocation probe changed from 48 B/call to 0 B/call. Timing at order 64 overlaps the noisy baseline, so this is
  an allocation fix rather than a claimed elapsed-time win.
- SYMV now groups four adjacent columns and composes the existing `dot4` and `axpy4` leaves over their common
  triangular run. The diagonal four-by-four part is handled directly, and the unselected triangle is never read.
  This reduces repeated `x` and `y` traffic without adding a public kernel or changing matrix storage.
- The 64 through 256 JVM sweep does not support the grouped traversal: it ranges from level to slower and can
  amplify Vector API compilation variability. Order 512 is the first retained boundary. In the paired JVM SIMD
  run, lower SYMV changed from 44.6 to 35.8 us/op and upper from 79.1 to 64.1. Scalar always keeps the original
  one-column traversal.
- At order 1024, two final built-in JVM passes measured lower SYMV at 127.6 and 130.1 us/op and upper at 122.1 and
  116.5. Their OpenBLAS counterparts were 82.7/85.9 and 83.3/84.2. At order 2048, stable built-in observations
  were 678.9 lower and 639.9/671.4 upper against OpenBLAS 471.6/476.1 and 481.1/476.8; one additional lower pass
  was contention-dominated and is retained. The remaining JVM gap is therefore workload- and run-dependent,
  commonly about 1.3 to 1.5 times OpenBLAS in the stable large cases rather than the roughly 1.1 objective.
- On Linux x86-64 Native, grouped lower SYMV changed from 46.9 to 33.5 us/op at 512, 168.2 to 132.0 at 1024 and
  921.4 to 818.5 at 2048. Upper changed from 48.8 to 35.4, 135.5 to 126.9 and 968.4 to 666.1. Native OpenBLAS
  varied between the adjacent passes; the post-change residual is roughly 1.2 to 1.6 times OpenBLAS at 1024 and
  2048, with the raw values retained.

## Raw files

- `jvm-main-baseline.json`: original current-main built-in/OpenBLAS GEMV and SYMV pass.
- `jvm-baseline-forced.json` and `jvm-symv4-forced.json`: independent explicit C/SIMD passes with the grouped
  traversal disabled and enabled at orders 256, 1024 and 2048.
- `jvm-crossover-baseline.json` and `jvm-crossover-symv4.json`: explicit C/SIMD crossover sweep.
- `jvm-512-baseline.json` and `jvm-512-symv4.json`: focused paired boundary measurements.
- `jvm-final-1.json` and `jvm-final-2.json`: final built-in/OpenBLAS operation passes.
- `jvm-gemv-allocation.json`: post-change order-64 run whose setup verified 0 B/call for all four dense Level 2
  allocation probes.
- `linuxX64-baseline.json` and `linuxX64-symv4.json`: Native compiled-in C/OpenBLAS comparison.

The benchmark output occasionally includes a decimal rendering of the setup checksum array. It is noisy console
formatting only; the raw JSON rows and resolved arm assertions are unaffected.
