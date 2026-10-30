<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

# Koblas

[![Maven Central](https://img.shields.io/maven-central/v/com.eignex/koblas.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/com.eignex/koblas)
[![Build](https://github.com/eignex/koblas/actions/workflows/build.yml/badge.svg)](https://github.com/eignex/koblas/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/eignex/koblas/branch/main/graph/badge.svg)](https://codecov.io/gh/eignex/koblas)
[![License](https://img.shields.io/github/license/eignex/koblas)](https://github.com/eignex/koblas/blob/main/LICENSE)

Dense and sparse double-precision BLAS Levels 1–3 for Kotlin Multiplatform, using `DoubleArray` storage.
Every supported operation works without a native library. JVM SIMD and optional native acceleration
are selected automatically. Factorizations and general solvers belong to consumers.

## Setup

Add the dependency to your JVM or `commonMain` source set:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported: JDK 17+, Android API 24+, Linux x64/arm64, Windows x64, macOS arm64, and iOS arm64 devices and simulators.
On desktop JDK 25+, enable SIMD with `--add-modules=jdk.incubator.vector`.
See [engine loading and native libraries](BACKENDS.md) for optional acceleration and setup.

## Kotlin

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.ofRows(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))
val y = a * x
```

Here `a * x` is a matrix-vector product. Operators return new results; calls ending in `Into` write
into existing arrays. See the [usage guide](USAGE.md) for more examples and reusing temporary storage.

## Guides

- [Engines and native libraries](BACKENDS.md): loading diagram, function coverage, thresholds and measured gains.
- [Usage](USAGE.md): vector and matrix operations, reusable arrays, views, and Java and mobile setup.
- [API overview](koblas/module.md): dense, sparse and raw array operations.
- [Benchmarks](koblas-bench/README.md): reproducible comparisons and measurement reports.

## Development

Run `./gradlew check lintDocs`. Use `-Pkoblas.noSimd=true` to withhold the JVM Vector API module;
installed native libraries remain visible.

On Windows, use `gradlew.bat check lintDocs` with Git Bash available for benchmark script tests.
