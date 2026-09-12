# Dense iamax investigation

`Vector.iamax()` scanned entries with `forEachStored` even for a contiguous dense vector. Unlike `dot`,
`asum`, and `sum`, it never reached the selected C or JVM SIMD kernels. The benchmark also called that
public extension, so selecting a different benchmark engine could not accelerate this operation.

The dense path now calls `DenseVectorKernels.iamax`. The portable scalar kernel defines the contract.
C and JVM SIMD first reduce magnitudes with four independent vector accumulators, then scan for the first
matching index. Strict comparisons ignore NaNs; the ordered search preserves first ties, including
infinities. Empty runs return -1, and runs containing only zeros and NaNs return 0. Offsets select a window,
and the returned index is relative to that window.

Four accumulators avoid a single maximum dependency chain. The C implementation names them individually:
GCC kept a vector array on the stack in the initial experiment, but keeps named accumulators in registers.
The second scan can reread the entire input, so the tradeoff depends on maximum position and working-set size.

## Measurements

Intel Core i9-12900H, JDK 25.0.1, Kotlin 2.4.10, deterministic uniform fixtures. JVM results use JMH with
five 200 ms warmup iterations, five measured iterations, and two forks. Native results use the suite's
calibrated runner with five samples and a 200 ms target. The table uses medians in ns/op, with speedups
relative to the scalar implementation of the same runtime. All table measurements ran on CPU 4, a
performance core, with fresh Gradle processes so the JMH forks inherited that affinity.

| Entries | JVM scalar | JVM SIMD | JVM C | Native scalar | Native C |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,213 | 842 (1.44x) | 935 (1.30x) | 4,328 | 1,501 (2.88x) |
| 65,536 | 19,647 | 11,687 (1.68x) | 13,404 (1.47x) | 65,994 | 15,065 (4.38x) |

The initial unpinned capture measured JVM scalar/SIMD at 1,236/656 ns for 4,096 entries (1.88x) and
18,228/9,484 ns for 65,536 entries (1.92x). Background CPU load rose during the follow-up. Pinning reduces
core migration but does not remove background contention or frequency changes. These are host-specific
observations, not universal speedup guarantees. The CPU traces and every retained sample are included.

SIMD starts at 256 entries. JVM C starts conservatively at 4,096: the initial 512-entry C cutoff lost to
scalar, and the 1,024-entry comparison changed direction between the unpinned and pinned runs. The native
binding uses its existing 48-entry C threshold. The expanded workload retains small cases to expose routing
costs as well as the larger reductions.

## Provenance

- `jvm-scalar.csv`, `jvm-simd.csv`, `native.csv`, and the vendor CSVs are the original capture at
  `22929a90abb13057cc58efa8b32da4ced1eedaf9`; the original C output is `jvm-c-cutoff-512.csv`.
- `jvm-c.csv` and `pinned-jvm-*.csv` use `79dcf45e12c8d40de824bd29b763a05960f2973b`, with a 1,024-entry
  JVM C cutoff. The final change only raises that cutoff to 4,096; the C and SIMD loop bodies in the table
  are identical. C timings below 4,096 in those files do not all represent the final routing policy.
- Both native scalar comparisons reuse the same saved native executable as `native.csv`, setting
  `KOBLAS_DENSE_NATIVE_C_MIN_LENGTH=1000001` to force the scalar fallback. Their engine label remains
  `native/c`; the override, rather than a different engine, selects the measured scalar implementation.
- `before-jvm-public.csv` is a separate pre-change JMH baseline at
  `1be759ff22bbd5a3e5c23b8983643fd46472a593`, using `before-cases.txt` and the same JVM sampling settings.
  It has no CPU trace. It measures the old public extension rather than the kernel interface.
- `cpu.csv`, `cpu-followup.csv`, and `cpu-pinned.csv` correspond to the execution intervals in
  `metadata.txt`. The vendor CSVs belong to the original unpinned capture.

## Reproduce

Run from the repository root:

```bash
koblas-bench/capture-report.sh --operation iamax --libraries openblas,onemkl \
  --warmups 5 --samples 5 --target-ms 200 --forks 2

# On this host CPU 4 is a performance core. Run each engine separately.
for engine in Scalar C Simd; do
  taskset -c 4 ./gradlew --no-daemon :koblas-bench:jvm${engine}Benchmark \
    -Pbench.operation=iamax -Pbench.warmups=5 -Pbench.samples=5 \
    -Pbench.targetMs=200 -Pbench.forks=2
done

# Native C, then the native scalar fallback, with the same CPU affinity.
taskset -c 4 ./gradlew --no-daemon :koblas-bench:nativeBenchmark \
  -Pbench.operation=iamax -Pbench.warmups=5 -Pbench.samples=5 -Pbench.targetMs=200 \
  -Pbench.output=/tmp/iamax-native-c.csv
KOBLAS_DENSE_NATIVE_C_MIN_LENGTH=1000001 taskset -c 4 ./gradlew --no-daemon \
  :koblas-bench:nativeBenchmark -Pbench.operation=iamax -Pbench.warmups=5 \
  -Pbench.samples=5 -Pbench.targetMs=200 -Pbench.output=/tmp/iamax-native-scalar.csv
```

Use `koblas-bench/tools/compare.sh --mode logical --timing arithmetic --require-compatible` with two JVM
CSVs, or with the two native CSVs. JVM and Native warmup timing boundaries differ, so compare speedups within
a runtime.
