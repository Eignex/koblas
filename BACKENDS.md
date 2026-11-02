# Engines and native libraries

You can use every supported operation without installing a native library. Koblas works with ordinary
Kotlin double arrays and chooses an implementation for you. On the JVM, enabling SIMD is the first step
for faster vector and matrix arithmetic. A native library can help with larger matrix operations.

Operators such as `a * b` and convenience functions use the default engine, available as `koblas`.

## How the default engine loads

The first use of `koblas` selects the engine once, using what is available on your platform:

| Platform | What the default engine uses |
|---|---|
| JDK 25+ | Owned JVM SIMD when enabled, plus available host libraries through the platform profile. |
| Kotlin/Native, macOS | Accelerate + portable Kotlin. |
| Kotlin/Native, Linux and Windows | oneMKL on x64, OpenBLAS, and portable Kotlin. |
| Other Kotlin/Native targets | Portable Kotlin. |
| JDK 17–24, Android and iOS | Portable Kotlin; no native library lookup. |

The default selects one complete prebuilt profile for its runtime, OS and architecture. macOS uses
Accelerate; Linux and Windows use oneMKL on x64 and OpenBLAS. JVM profiles also include owned SIMD and
check AOCL on Linux/Windows x64 or ArmPL on Arm64. Each vendor has its own routing rules; unavailable
libraries, exports and optional modules fall through to the remaining providers and portable coverage.
Without native-access permission the JVM selects its owned SIMD or portable profile.

Vendor cutoffs are provisional estimates pending measurement before this PR merges. Existing oneMKL
numbers supply its starting policy; OpenBLAS, Accelerate, AOCL and ArmPL have separate estimates.
These profiles do not claim measured applicability across CPU models. Owned SIMD detects the available
Vector API width. Explicit locally calibrated profiles can restrict applicability to an exact CPU model.

On JDK 25+, enable SIMD with this JVM argument:

```text
--add-modules=jdk.incubator.vector
```

SIMD processes several numbers in one CPU instruction. It requires no native numerical library and
works independently of native access.

On x86-64, vector-heavy workloads can also add `-XX:ObjectAlignmentInBytes=32`. HotSpot then starts
every array at a known offset, so dot products, absolute sums and maximum-magnitude searches can align
their loads. On Intel AVX-512 lab hosts, with both settings on the same machine
([10006](https://lab.eignex.com/jobs/10006), [10007](https://lab.eignex.com/jobs/10007)), dot products
of 4,096 elements and longer ran 13–19% faster, and norms and AXPY of 4,096 to 16,384 elements 21–28%
faster, and absolute sums 4–16% faster. AXPY of 65,536 elements ran 3–11% slower. Scaling, swaps,
rotations and vectors shorter than 4,096 elements changed by a few percent either way. The cost is memory:
every object is padded to 32 bytes.

## When a native library helps

On the JVM, vector operations and matrix-vector operations stay in Kotlin. SIMD covers dot products,
sums, norms, absolute sums, maximum-magnitude searches and scaled vector addition (AXPY), with scalar
fallbacks. Scaling, copying, swapping and rotations use Kotlin loops or array-copy intrinsics.
Matrix-vector products and rank updates use SIMD where applicable. Sparse operations keep their Kotlin
traversal and can use SIMD for the arithmetic within it.

Automatic JVM native calls are limited to larger matrix-matrix operations: GEMM, GEMMT, SYMM, SYRK,
SYR2K, TRMM and TRSM. The provisional minimum output and shared dimensions start at 64 for oneMKL
and Accelerate, or 128 for OpenBLAS, AOCL and ArmPL; GEMMT and borrowed views require twice those
extents. oneMKL GEMM also qualifies with at least 16 output rows and columns and a shared dimension
of 1,024–4,096. Kotlin/Native has separate vector and work cutoffs for each vendor.

JVM native calls copy arrays to and from native memory. An Intel comparison that includes this
cost measured oneMKL at 1.45–4.66x JVM SIMD throughput for matrix-matrix cases at orders 128, 256
and 768. For those workloads, installing oneMKL can pay off. For vector and matrix-vector workloads,
enabling SIMD is enough to reach the default engine's accelerated paths. These figures come from Intel
hosts; measure before relying on them elsewhere. Kotlin/Native can pin arrays in place and uses
separate native cutoffs.

To allow native calls on JDK 25+, also pass `--enable-native-access=ALL-UNNAMED` on the classpath,
or `--enable-native-access=com.eignex.koblas` on the module path. Install a compatible library using 32-bit BLAS
indices (LP64) and make it discoverable by the system loader. Koblas also searches `~/intel/oneapi/mkl/latest` and
`/opt/intel/oneapi/mkl/latest` for oneMKL on Linux. Set flags and library locations before starting the application.

Explicit bindings support Accelerate, OpenBLAS, AOCL, ArmPL and oneMKL on their supported hosts.
On Windows, place matching LP64 DLLs and their dependencies in an absolute directory on PATH. The manual
factory `BuiltinEngines.vendor(vendor)` returns an engine without changing the default.
Use `BuiltinEngines.scalar` to select portable Kotlin explicitly.

## Explicit composition and recalculation

`KoblasEngine` is the single contract for portable, SIMD, vendor and combined engines. Implementations
implement the parts they supply and declare positive `EngineImplementation` sets. The interface contains
no arithmetic defaults: unsupported parts fail without calling another engine. `PortableEngine` owns the
portable arithmetic, and vendor providers contain only their foreign calls and binding adaptations. `BuiltinEngines` selects exact portable, SIMD or named vendor engines. Vendor
capabilities follow exported symbols, and the composer binds capabilities to their actual implementations.
The combined engine explicitly calls a separate portable provider for unselected operations. Shared schedules still own dense
traversal, CSC traversal, validation and alias staging; packed product geometry is selected before packing.

```kotlin
val profile = StandardEngineProfiles.forTarget(EngineRuntime.Native, hostPlatform())
val engine = BuiltinEngines.configured(profile)
```

Profiles contain ordered `EngineRule` preferences: unconditional selection, length/work thresholds, and
bounded shape/layout regions. For an explicit profile, check `profile.appliesTo(runtime, hostPlatform())`.
To load a recalculated profile, pass its text to `EngineProfile.parse`; the library performs no file I/O.
`BuiltinEngines.configured` resolves named libraries and retains owned SIMD preferences beneath custom
rules. Missing optional libraries fall through; unknown source identifiers are rejected. Use
`KoblasEngine.combine(profile, providers...)` for explicit custom providers or scalar-only composition.
Panel, product and triangular families require unconditional rules so the
schedule and its packed arithmetic agree on geometry. Intrinsic scalar edges and safe norm retries remain
inside their implementations and are reflected by internal implementation attribution. The application engine
uses shared checked operations for dense and sparse scheduling; separate backend and BLAS interfaces, vendor adapters, vendor preference selection and route objects are
removed. Explicit composition flattens the supplied engines into arithmetic sources once, with intrinsic
portable coverage. The composer binds each arithmetic family once, and forwards its calls to that bound provider. No engine inherits implementation code.

Recalculate a profile explicitly on the intended host:

```bash
python3 koblas-bench/tune-engine.py --capture --runtime native --backend onemkl \
  --output /tmp/native-onemkl-profile
```

The workflow retains three paired rounds, timing metadata, CSVs, ratios and hashes, then emits `profile.tsv`.
On Linux, fitting and validation share one automatically selected allowed core, recorded in each capture.
It uses twenty warmups of 200 ms and validates the combined configuration on disjoint holdout cases.
`validation.json` records the dispatcher-selected providers and all paired ratios. Persistent regressions
over five percent in either selected routes or unchanged controls fail validation; missing, empty or
untimed validation cases fail as well. Profile names include a hash of their configuration and evidence.
Contiguous and strided length thresholds are fitted independently and must keep winning at every larger measured size; matrix preferences are bounded to
measured extents and require matching variants to agree. Unmeasured regions retain portable coverage.
Use `--runtime jvm` for transfer-inclusive JVM vendor tuning or `--backend jvm-simd` for owned vector tuning.
SIMD fitting forces its supplied vector bodies so an existing cutoff cannot hide a better crossover;
packing families retain their owned configuration. Capability metadata filters unsupported operations.
The emitted profile uses the library's runtime CPU identity from capture metadata, so it matches startup
applicability checks. Mixed models require separate captures. Set `--os` and `--arch` for targets other
than Linux x64. If the runtime cannot detect the model, explicitly use `--cpu '*'` for an OS and
architecture profile; hardware display names remain in the capture as evidence.
JVM vendor fitting compares against the owned SIMD engine by default; `--baseline jvm-scalar` supports deployments without SIMD.

Validate the emitted configuration on additional cases and compare the combined engine, not only direct
vendor arms. `KOBLAS_BENCH_ENGINE_PROFILE=/absolute/path/profile.tsv` selects it on the benchmark harness's
`native-default` or `jvm-default` arm, without changing the library's process default. Refit existing repeated
captures with `--results PATH` instead of `--capture`. SIMD captures need current capability metadata.
Use `--candidate native-default` with an explicitly forced profile to fit combined borrowed-layout captures.
Timing smoke runs cannot produce profiles.

Benchmark reports and result data stay outside the source tree. The retained lab reports cover
[Native calibration](https://lab.eignex.com/jobs/10142) and the [JVM comparison](https://lab.eignex.com/jobs/10183).
Use the maintained client, `lab fetch ID /tmp/report`, to retrieve their CSVs, metadata and hashes.

See the [API overview](koblas/module.md) for explicit engines and bindings, or the
[usage guide](USAGE.md) for array, workspace and view examples.
