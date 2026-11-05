# Engines and native libraries

Koblas supports dense and sparse BLAS Levels 1–3 without a native library. Convenience functions
and libraries using `koblas` share one immutable engine, selected once on first use.

## Choose the default engine

Most applications need no configuration. To override the default programmatically, call once at
application startup, before any library or convenience operation uses `koblas`:

```kotlin
import com.eignex.koblas.configureKoblas
import com.eignex.koblas.engine.BuiltinEngines

fun main() {
    configureKoblas(engine = BuiltinEngines.scalar)
    // Start your application and libraries here.
}
```

From Java, use `Koblas.configureKoblas(BuiltinEngines.getScalar())`.
Configuring twice, or after selection has started, throws `IllegalStateException`.
The supplied engine is used as is; exact vendor engines can have unsupported operations.

Alternatively, set `KOBLAS_ENGINE=scalar` before launching the application. On JVM, use
`-Dkoblas.engine=scalar`. Precedence is programmatic configuration, JVM system property,
environment variable, then automatic selection. Launch settings are read on first engine use.

| Setting | Selection |
|---|---|
| `auto` | Automatic platform selection, also used when no setting is supplied. |
| `scalar` | Portable Kotlin without SIMD or host library lookup. |
| `simd` | Owned JVM SIMD kernels with portable fallbacks. Requires the Vector API module. |
| `accelerate`, `onemkl`, `openblas`, `aocl`, `armpl` | The named vendor for its exported operations, with portable coverage for the rest. |

Settings ignore case and surrounding whitespace. Unknown names and unavailable explicit selections
fail on first use. Vendor selections require a compatible library and native-access permission on JVM;
they bypass automatic crossover thresholds. `auto` retains fallbacks when optional providers are unavailable.
Direct calls to other engines from `BuiltinEngines` are unaffected by startup configuration.

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

## Exact engines and tuning

`BuiltinEngines.scalar` selects portable Kotlin; `BuiltinEngines.simd` and
`BuiltinEngines.vendor(Vendor.OneMkl)` return an exact engine or null when unavailable.
An exact vendor engine supports only its declared operations. To combine providers with portable coverage:

```kotlin
val profile = StandardEngineProfiles.forTarget(EngineRuntime.Native, hostPlatform())
val engine = BuiltinEngines.configured(profile)
```

A custom profile's applicability can be checked with `profile.appliesTo(runtime, hostPlatform())`; include SIMD rules explicitly when wanted.
Profiles are recalculated explicitly through the [benchmark workflow](koblas-bench/REFERENCE.md#recalculate-routing-profiles);
application startup only selects a prebuilt one.
