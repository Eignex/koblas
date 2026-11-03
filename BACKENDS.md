# Engines and native libraries

Koblas supports dense and sparse BLAS Levels 1–3 without a native library. Operators such as `a * b`
and convenience functions use `koblas`, an immutable engine selected once on first use.

## Automatic selection

The default combines available implementations using one standard profile for the runtime and target.
JDK 25+ adds Kotlin SIMD when enabled; JDK 17–24, Android and iOS use portable Kotlin without host lookup.
Missing libraries, exports or native-access permission retain the remaining providers and portable coverage.

| Target | Standard composition and tuning |
|---|---|
| Linux x86-64 | oneMKL, OpenBLAS and portable Kotlin have separate AVX2/AVX-512 tunings. JVM profiles add SIMD and estimated AOCL preferences. |
| Linux Arm64 | OpenBLAS and portable Kotlin use SVE-256 tuning. JVM profiles add SIMD and estimated ArmPL preferences. |
| macOS Arm64 | Accelerate combines with portable Kotlin, plus SIMD on JVM. Native thresholds use measurements; JVM thresholds are estimates. |
| Other supported hosts | Windows borrows Linux tuning; Intel macOS borrows Arm64 Accelerate and Linux x86 JVM SIMD tuning. Other CPU classes and SVE widths use the closest compatible profile, labeled unmeasured best effort. |

Profiles contain broad operation, shape and layout thresholds. Native routing favors host libraries;
JVM transfers use higher cutoffs. Declarations and measurement references live in
`engine/StandardEngineProfiles.kt`; thresholds are tuning estimates.

## Enable acceleration

On JDK 25+, enable SIMD with the first flag and native library calls with the second:

```text
--add-modules=jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
```

On the module path, use `--enable-native-access=com.eignex.koblas` instead.
Install compatible LP64 libraries with 32-bit BLAS indices and make them discoverable by the system loader.
Windows DLLs and dependencies need an absolute directory on PATH. Linux oneMKL is also searched under
`~/intel/oneapi/mkl/latest` and `/opt/intel/oneapi/mkl/latest`. Set these before application startup.

JVM native calls copy arrays to and from native memory, so they mainly help larger matrix products.
Kotlin/Native can pin arrays in place. Sparse operations keep portable CSC traversal and use selected
kernels for arithmetic within it.

## Explicit engines

`BuiltinEngines.scalar` selects portable Kotlin; `BuiltinEngines.simd` and
`BuiltinEngines.vendor(Vendor.OneMkl)` return an exact engine or null when unavailable.
An exact vendor engine supports only its declared operations. To combine providers with portable coverage:

```kotlin
val profile = StandardEngineProfiles.forTarget(EngineRuntime.Native, hostPlatform())
val engine = BuiltinEngines.configured(profile)
```

Use `KoblasEngine.combine(profile, providers...)` for custom providers. A custom profile's applicability
can be checked with `profile.appliesTo(runtime, hostPlatform())`; include SIMD rules explicitly when wanted.

All implementations share `KoblasEngine` and declare positive `EngineImplementation` sets. Composition
selects arithmetic through profile rules; shared operations handle validation, aliases and scheduling.
Portable Levels 1–3 live under `internal/portable/dense/` and `internal/portable/sparse/`; vendor engines
remain independent.

## Recalculate profiles

The [benchmark workflow](koblas-bench/REFERENCE.md#recalculate-routing-profiles) captures paired workloads,
emits Kotlin configuration and validates rebuilt profiles on disjoint holdouts. Recalculation is explicit;
application startup only selects a prebuilt profile. Keep reports and measurement data outside the repository.
