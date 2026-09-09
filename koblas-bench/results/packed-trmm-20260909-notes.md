# Packed TRMM benchmark notes

The runs in `packed-trmm-20260909.tar.gz` exercise the packed TRMM implementation based on
`d3148d5a66dfcc998182e27a1363e0c8dcf2cf8b9` plus the uncommitted change under review. They ran on an
Intel Core i9-12900H with affinity `0,2,4,6`. Other benchmark and build processes were active on this shared
host, so every raw confidence interval is retained and noisy point estimates are not treated as crossovers.

## Coverage

The two independent JVM comparison passes cover orders 16, 32, 64, 128 and 257; narrow, medium and wide
panels; left and right multiplication; lower and upper storage; transposed and non-transposed operations;
and explicit and unit diagonals. Each pass compares the built-in Vector API implementation with a
benchmark-owned, single-threaded OpenBLAS binding. The Kotlin/Native pass covers orders 16, 64 and 128 and
the same six flag combinations with the compiled-in C tiles and native OpenBLAS binding.

The forced JVM runs separately compare packed and reference traversals around order 16, panels 31, 32 and
33, and at 64x32 and 128x256. The `koblas.noSimd` runs omit the Vector module and therefore resolve to the
portable C/scalar tile rather than accidentally retaining SIMD. oneMKL 2026.1 was requested during
development but was unavailable on this host, so no oneMKL result is claimed.

Every case includes copying the original input into the mutable result. The built-in arm also includes its
required immutable snapshots and packed panels. Packing-only cost remains independently visible in
`PackedPanelBenchmark`; it is not subtracted from the TRMM timings.

## Findings

- The forced crossover sweep does not support packing at order 15 or panel 31. Order 16 and panel 32 are
  the first conservative, tile-aligned gate: results there range from parity to a clear win depending on
  orientation, while both measured orientations win at panel 33 and order 17.
- At 64x32 the packed and reference JVM paths overlap within host variability. This is a parity region, not
  a claimed speedup.
- At 128x256, representative forced JVM SIMD measurements are 459 and 469 us/op packed versus 705 and
  672 us/op reference. The no-SIMD large measurements are 959 and 888 us/op packed versus 1696 and
  2557 us/op reference, with wide confidence intervals retained in the raw files.
- Across the full final JVM pass, 128x256 built-in results span roughly 310--807 us/op as contention varies;
  OpenBLAS spans roughly 173--237 us/op. The implementation materially narrows the old reference gap but
  does not reach the external comparator.
- Kotlin/Native built-in C measures roughly 4--8 us/op at 16x32, 39--54 us/op at 64x32 and
  445--715 us/op at 128x256. Native OpenBLAS remains about 0.5--0.7, 5--6 and 134--216 us/op respectively.

## Raw files

- `jvm-pass-1.json` and `jvm-pass-2.json`: independent final SIMD/OpenBLAS comparison passes.
- `jvm-boundaries-packed.json` and `jvm-boundaries-reference.json`: forced order and panel crossover sweep.
- `jvm-simd-packed.json` and `jvm-simd-reference.json`: representative packed/reference SIMD scale sweep.
- `jvm-portable-packed.json`, `jvm-portable-reference-boundary.json` and
  `jvm-portable-reference-large.json`: no-SIMD packed/reference evidence.
- `linuxX64.json`: Kotlin/Native built-in C and native OpenBLAS comparison.

The Gradle benchmark configuration uses three 500 ms warmups and five 500 ms measurements per JVM case.
Kotlin/Native uses the corresponding five-sample harness. The global JMH lock check was disabled because
another benchmark process held the shared host lock; this does not disable measurement isolation inside
each fork.
