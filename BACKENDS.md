# Engine selection

Koblas initializes a shared engine from launch settings or automatic platform selection. Libraries and
convenience operations use it through `koblas`. Most applications need no setup.

## Override at startup

Choose an engine during startup, before libraries use Koblas or concurrent work begins:

```kotlin
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas

fun main() {
    koblas = KoblasEngine.scalar
    // Start the application and its libraries.
}
```

Launch settings accept the names below, for example `KOBLAS_ENGINE=scalar` or, on JVM, `-Dkoblas.engine=scalar`.
Java applications can call `Koblas.setDefault(KoblasEngine.getScalar())`.

| Name | Selection |
|---|---|
| `auto` | Platform tuning with available acceleration and portable fallback. This is the default. |
| `scalar` | Portable Kotlin, without SIMD or host library lookup. |
| `simd` | Owned JVM Vector API kernels with portable fallbacks. Requires JDK 25+ and the Vector API module. |
| `accelerate`, `onemkl`, `openblas`, `aocl`, `armpl` | The named vendor, with portable coverage for operations it does not export. |

The initial default uses the JVM property, then the environment variable. Assigning `koblas` directly
replaces that initial value with any `KoblasEngine`, used as supplied.
The initial default is constructed before a programmatic override, so launch settings must be valid
even when overridden. Names ignore case and surrounding whitespace; unknown or unavailable selections
fail when resolved.

Reads and configuration are unsynchronized. Callers own startup order and safe publication before
concurrent use; the library does not guard against late or concurrent configuration.

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
exports use the owned SIMD/scalar kernels. The cutoffs account for whether JVM SIMD is available.
Without it, x64 OpenBLAS handles contiguous `asum` and `iamax` reductions from 4,096 through 65,536
elements. Shorter calls and larger streaming vectors stay scalar because native-memory transfers cost
more than the faster reduction saves. JVM matrix–vector calls retain the owned schedule: transfer-inclusive
measurements have not established a broad vendor crossover. These are internal cutoffs, not user settings.
Explicitly naming a vendor uses it for every operation it exports. JDK 17–24, Android and iOS automatically
use portable Kotlin.
Sparse operations always keep portable CSC traversal. See the [benchmark reference](koblas-bench/REFERENCE.md)
for tuning and measurement details.

## Exact engines for comparisons

`KoblasEngine.scalar`, `KoblasEngine.simd` and `KoblasEngine.vendor(Vendor.OneMkl)` select exact
engines for tests and benchmarks. Optional engines return null when unavailable. An exact vendor engine
supports only its exported operations; launch selection by name includes portable coverage.
Calls through an explicit engine instance are independent of the shared default.
