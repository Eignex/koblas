# Native capability smoke evidence

These are short PR 02 build-transition samples, not threshold calibration. `hardware.txt` is the output of
`koblas-bench/tools/hardware.sh`; its SHA-256 names the parent directory. CSV run records retain exact source
SHAs, dirty state, runtime and timing boundaries. The before source is `bb25cc5f`; the after/raw source is
`79ec551f`. Later macOS build fixes do not change the measured arithmetic or JVM bindings.

Each runtime uses the same `dot+4096+uniform` case, two warmups, three measured samples, 50 ms targets and one
JVM fork. JVM and Native have different harnesses and are compared only with their own before sample. The
machine was not reserved; other build activity and CPU scheduling can affect these very short samples.
The JVM policy medians are 800.9 ns before and 670.0 ns after; Native medians are 978.7 ns and 521.4 ns.
These observations do not establish a speedup or justify a policy change.

Run from the repository root, with this directory substituted for `$report`:

```sh
./gradlew :koblas-bench:jvmCBenchmark :koblas-bench:nativeBenchmark \
  -Pbench.operation=dot -Pbench.cases="$report/cases.txt" \
  -Pbench.warmups=2 -Pbench.samples=3 -Pbench.targetMs=50 -Pbench.forks=1
./gradlew :koblas-bench:jvmCRawBenchmark -Pbench.variant=avx2 \
  -Pbench.operation=dot -Pbench.cases="$report/cases.txt" \
  -Pbench.warmups=2 -Pbench.samples=3 -Pbench.targetMs=50 -Pbench.forks=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.variant=avx2 \
  -Pbench.operation=dot -Pbench.cases="$report/cases.txt" \
  -Pbench.warmups=2 -Pbench.samples=3 -Pbench.targetMs=50
```

Repeat the exact modes with `scalar` and `sse2`. Their case records name the implementation symbols;
the operation-specific dot IDs are scalar 17, SSE2 18 and AVX2 19. Ordinary packed-four inputs use layout
`0x10001` and column-major-four output uses `0x20001`; the dot samples have no packed operands.

`toolchain-*.txt` records compiler/target/flags for the shared object and both Native archives. The assembly
files retain dot, entry validation and probe/initializer excerpts from those objects, using `objdump -d`
(the Kotlin/Native target's `aarch64-unknown-linux-gnu-objdump` for Arm). Scalar arithmetic uses scalar
instructions, SSE2 uses 128-bit registers, AVX2 uses 256-bit registers, and NEON uses 128-bit registers.
The probe and entry validation remain baseline code. Wider source vectors lower to multiple SSE2/NEON
registers; logical batch size is independent of register width.

The accompanying logs preserve ABI/export, sanitizer, baseline x86 emulation, Arm64 emulation, and JVM
allocation/concurrency evidence. Arm64 emulation ran all 505 Native tests on Cortex-A53; it is not physical
Arm execution or performance evidence. macOS hardware execution and final review/CI acceptance are recorded
in the single PR handoff, not inferred from cross-compilation or these samples.

`linux-arm64-shared-*.log` additionally records a Linux Arm shared-library cross-build with the selected
Native compiler/sysroot, C fixture dynamic linking, all 25 ABI exports, and successful ABI/guard execution
under Cortex-A53 emulation. Linux shared builds explicitly restrict exports to the declaration-header ABI
because older cross-linkers otherwise expose CRT boundary symbols. This check still does not establish
physical Linux Arm execution or NEON performance; those remain outstanding in the PR handoff.
