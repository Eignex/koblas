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

The standard profiles use broad thresholds informed by retained measurements, with Native routing
favoring host libraries; JVM transfers use higher cutoffs. These are tuning estimates, not performance guarantees.

| Target | Standard tuning |
|---|---|
| Linux x86-64 | Separate AVX2 and AVX-512 profiles combine oneMKL, OpenBLAS and portable; JVM adds SIMD and estimated AOCL preferences. Tunings use [Native AVX2](https://lab.eignex.com/jobs/10583), [Native AVX-512](https://lab.eignex.com/jobs/10565), [JVM AVX2](https://lab.eignex.com/jobs/10584) and [JVM AVX-512](https://lab.eignex.com/jobs/10566) measurements. |
| Linux Arm64 | SVE-256 profiles combine OpenBLAS and portable, with SIMD and estimated ArmPL preferences on JVM. Tunings use [Native](https://lab.eignex.com/jobs/10628) and [JVM](https://lab.eignex.com/jobs/10607) measurements. |
| macOS Arm64 | Accelerate combines with portable, plus SIMD on JVM. Native tuning uses [measurements](https://lab.eignex.com/jobs/10630); JVM thresholds are estimates. |
| Other supported hosts | Windows borrows Linux tuning; Intel macOS borrows Accelerate tuning from Arm64 and JVM SIMD tuning from Linux x86. Other CPU classes and SVE widths borrow the closest compatible profile, explicitly labeled unmeasured best effort. |

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

The platform profile owns each vendor's operation, shape, work and layout cutoffs. The embedded
configuration files contain these preferences; Kotlin/Native and JVM bindings have separate tunings.

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
no arithmetic defaults: unsupported parts fail without calling another engine. `PortableEngine` supplies
vector, indexed, panel, product and substitution arithmetic to the shared schedules. Vendor providers
contain only their foreign calls and binding adaptations. `BuiltinEngines` selects exact portable, SIMD
or named vendor engines. Vendor
capabilities follow exported symbols, and the composer binds capabilities to their actual implementations.
The combined engine explicitly calls a separate portable provider for unselected operations. Shared schedules still own dense
traversal, CSC traversal, validation and alias staging; packed product geometry is selected before packing.

```kotlin
val profile = StandardEngineProfiles.forTarget(EngineRuntime.Native, hostPlatform())
val engine = BuiltinEngines.configured(profile)
```

Profiles contain ordered `EngineRule` preferences: unconditional selection, length/work thresholds, and
bounded shape/layout regions. For an explicit profile, check `profile.appliesTo(runtime, hostPlatform())`.
Measured profiles can require a CPU instruction class (`Avx2`, `Avx512`, `Arm64` or `Sve`) independently of an exact model.
SVE profiles also require their measured active vector width. Native reads `PR_SVE_GET_VL`; JVM reads
HotSpot's active `UseSVE`/`MaxVectorSize` settings, including on the scalar arm without the Vector API.
Unknown or different widths do not match.
Standard selection borrows compatible preferences within the same runtime when no target profile matches,
and labels the result `best-effort-*` with its donor names. Unsupported vendors are excluded.
Standard profiles are typed Kotlin declarations in `engine/TargetProfiles.kt`; the library has no profile parser.
Portable vector, panel, product and triangular kernels live in `internal/portable/dense/` and
`internal/portable/sparse/`; `PortableEngine` forwards to them. Shared blocking and CSC traversal live in
`internal/dense/` and `internal/sparse/`, and call the selected engine or the fixed portable sparse kernels.
`BuiltinEngines.configured` resolves named libraries and uses exactly the profile's preferences.
Include owned SIMD rules explicitly when wanted. Missing optional libraries fall through; unknown source identifiers are rejected. Use
`KoblasEngine.combine(profile, providers...)` for explicit custom providers or scalar-only composition.
Panel, product and triangular families require unconditional rules so the
schedule and its packed arithmetic agree on geometry. Intrinsic scalar edges and safe norm retries remain
inside their implementations and are reflected by internal implementation attribution. The application engine
uses shared checked operations for dense and sparse scheduling; separate backend and BLAS interfaces, vendor adapters, vendor preference selection and route objects are
removed. Explicit composition flattens the supplied engines into arithmetic sources once, with intrinsic
portable coverage. The composer binds each arithmetic family once; shared schedules call that provider directly. No engine inherits implementation code.

Recalculate a profile explicitly on the intended host:

```bash
./gradlew :koblas-bench:prepareLabCalibration :koblas-bench:tuneEngine \
  --args="--capture --runtime native --backend onemkl --backend openblas --cpu * --native-executable koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --output /tmp/native-profile"
```

The Kotlin tool emits `Profile.kt` and external evidence. Use its declaration in the target's standard
configuration, rebuild, and capture disjoint holdouts using the ordinary `native-default` or `jvm-default` arm.
Use `--runtime jvm` for transfer-inclusive tuning, or repeat `--results PATH` instead of `--capture` to refit
fetched lab jobs; retain `lab show ID` and `lab execution ID` snapshots beside their results.
`--fallback aocl=openblas` and `--fallback armpl=openblas` explicitly mark unmeasured vendor estimates.

Check the rebuilt composition using its recorded generated profile name:

```bash
./gradlew :koblas-bench:tuneEngine \
  --args="--validate-results /tmp/holdouts --training /tmp/native-profile --runtime native --output /tmp/validated-profile"
```

Validation checks the compiled profile identity, disjoint cases, provider reach and paired ratios.
`--validation-cases PATH` selects an explicit holdout list; `generateCalibrationGrids` emits the standard grids.
Reports and measurement data stay outside the repository. Thresholds are broad routing preferences.

See the [API overview](koblas/module.md) for explicit engines and bindings, or the
[usage guide](USAGE.md) for array, workspace and view examples.
