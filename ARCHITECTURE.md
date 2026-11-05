# Architecture

Koblas keeps public storage and operation contracts in `com.eignex.koblas`, exact engine access and vendor metadata in
`com.eignex.koblas.engine`, and execution details under `internal`. All of these belong to the same
multiplatform module; platform source sets supply their own runtime and arithmetic implementations.

## Public entry points

- Containers, factories, views, serialization and materialized transpose describe or transform storage.
- `KoblasEngine` owns explicit BLAS calls, including windowed `ger` and sparse `dotColumn`. Its default
  matrix methods prepare their operands and invoke shared schedules. Primitive Level 1 methods are implemented
  by engines.
- `VectorOps.kt` is the small container facade over the immutable default engine in `DefaultEngine.kt`.
- `configureKoblas` selects the shared engine at startup by name or instance; launch settings use the same selector.
- `engine` exposes exact built-in engines and vendor metadata.

## Internal ownership

| Package | Owns |
|---|---|
| `storage` | Bounds, pooled arrays, physical windows and structural CSC utilities. |
| `operations` | Reusable BLAS validation, alias snapshots, scratch loans and vector storage adaptation. |
| `scheduling.dense` | Panels, packing, product blocking, selected triangles and triangular traversal. |
| `scheduling.sparse` | CSC traversal, right-hand-side grouping and sparse accumulation schedules. |
| `kernels` | The `EngineKernels` arithmetic contract, `PanelWork` and shared kernel policies. |
| `kernels.scalar` | The complete scalar engine and numerical leaves over addressed windows. |
| `engine` | Startup selection, fixed target tuning, host facts, provider composition and whole-call routing. |
| `host` | Shared host-library contracts and ABI adaptation. |

A default matrix method prepares its operands and calls a shared schedule, which invokes the selected
kernel provider. Preparation settles shapes, empty cases, BLAS no-read rules and aliasing before
arithmetic writes the destination. A composed engine can route an eligible whole call to a host engine;
that binding owns its argument preparation and numerical library call.

Schedules own traversal and workspace extents. Kernels own arithmetic and recommend grouping or packing
geometry through their contract. The schedule can process a smaller group; group width, register tile,
cache block and diagonal block are independent quantities. Scalar and SIMD engines share the schedules.

Scalar leaves depend on storage and kernel contracts. They do not invoke operation preparation or shared
schedules. Sparse traversal remains shared CSC traversal regardless of which Level 1 engine is selected.

## Platforms

JVM Vector API kernels remain under `jvmJava25/internal/jvm25/simd`. JVM and native host bindings remain
in their platform source sets under `internal/host`. Platform selection uses `internal/engine` actuals.
Klause and Kumulant consume the public engine and containers; platform implementation and build setup
stay within Koblas.

Tests mirror the internal package that owns the implementation. Public API and published-artifact
checks exercise calls through the engine and container entry points.

Tuning cutoffs live in `internal/engine/EngineTuning.kt`. Runtime and host facts select fixed tables;
unmeasured targets borrow compatible estimates. Applications do not construct routing rules or profiles.
The benchmark fitter emits internal tuning declarations and keeps target provenance in its evidence file.
