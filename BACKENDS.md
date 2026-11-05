# Engine selection

Koblas selects an engine automatically on first use. Libraries and convenience operations using `koblas`
share that immutable engine. Most applications need no setup.

## Override at startup

Choose an engine once, before any library or convenience operation uses Koblas:

```kotlin
import com.eignex.koblas.configureKoblas

fun main() {
    configureKoblas("scalar")
    // Start the application and its libraries.
}
```

Use the same names with `KOBLAS_ENGINE=scalar` or, on JVM, `-Dkoblas.engine=scalar`.
Java applications can call `Koblas.configureKoblas("scalar")`.

| Name | Selection |
|---|---|
| `auto` | Platform tuning with available acceleration and portable fallback. This is the default. |
| `scalar` | Portable Kotlin, without SIMD or host library lookup. |
| `simd` | Owned JVM Vector API kernels with portable fallbacks. Requires JDK 25+ and the Vector API module. |
| `accelerate`, `onemkl`, `openblas`, `aocl`, `armpl` | The named vendor, with portable coverage for operations it does not export. |

Programmatic configuration takes precedence over the JVM property, then the environment variable.
Names ignore case and surrounding whitespace. Unknown or unavailable selections fail on first use;
configuring twice or after selection has started throws `IllegalStateException`.
An existing `KoblasEngine` can also be passed to `configureKoblas` and is used as supplied.

## Enable acceleration

On JDK 25+, enable SIMD and native calls with:

```text
--add-modules=jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
```

On the module path, use `--enable-native-access=com.eignex.koblas` instead.
Vendor calls need compatible LP64 libraries with 32-bit BLAS indices, discoverable by the system loader.
Windows DLLs and their dependencies need an absolute directory on PATH. Linux oneMKL is also searched
under `~/intel/oneapi/mkl/latest` and `/opt/intel/oneapi/mkl/latest`.

Automatic selection tries compatible host libraries in platform order and keeps the first available vendor.
That library handles calls above internal measured cutoffs or documented estimates; other calls and missing
exports use the owned SIMD/scalar kernels. Explicitly naming a vendor uses it for every operation it exports. JDK 17–24, Android and iOS automatically use portable Kotlin.
Sparse operations always keep portable CSC traversal. See the [benchmark reference](koblas-bench/REFERENCE.md)
for tuning and measurement details.

## Exact engines for comparisons

`BuiltinEngines.scalar`, `BuiltinEngines.simd` and `BuiltinEngines.vendor(Vendor.OneMkl)` select exact
engines for tests and benchmarks. Optional engines return null when unavailable. An exact vendor engine
supports only its exported operations; startup selection by name includes portable coverage.
Calls through an explicit engine instance are independent of the shared default.
