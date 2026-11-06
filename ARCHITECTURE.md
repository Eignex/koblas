# Architecture

Koblas keeps public storage and operation contracts in `com.eignex.koblas`, engine configuration in
`com.eignex.koblas.engine`, and execution details under `internal`. All of these belong to the same
multiplatform module; platform source sets supply their own runtime and arithmetic implementations.

## Public entry points

- `Matrix` and `Vector` own fixed-shape mutable entry contracts and default materialization. `DenseMatrix`
  represents compact buffers and borrowed views with offsets and leading dimensions. `DenseVector` adds
  vector addressing; sparse setters maintain sorted structure and retain written zeros. Dense matrix equality
  and serialization visit logical entries only. Engine eligibility checks layout rather than concrete types.
- `KoblasEngine` owns typed BLAS calls over dense and sparse containers. Logical dimensions are validated
  before views are unpacked. Its matrix methods prepare their operands and invoke shared schedules.
  Primitive array windows, including windowed `ger`, belong to the internal `EngineKernels` interface.
- `VectorOps.kt` and `MatrixOps.kt` are small container facades over the immutable default engine in
  `DefaultEngine.kt`. Both facades share the `Norm` enum, with norm dispatch and validation under
  `internal/operations`. Every facade extension targets `Matrix` or `Vector`; copies and matrix
  transposes are storage transformations, and copy-into operations validate shapes and stage aliases.
  Sparse reductions traverse stored entries; custom storage uses scalar readers. Vector scale and AXPY
  use the same storage adapters with the default engine. Explicit BLAS calls stay on the engine.
- `engine` owns exact built-in engines, profiles, platform facts and provenance.

## Internal ownership

| Package | Owns |
|---|---|
| `storage` | Bounds, pooled arrays, physical windows and structural CSC utilities. |
| `operations` | Reusable BLAS validation, alias snapshots, scratch loans and vector storage adaptation. |
| `scheduling.dense` | Panels, packing, product blocking, selected triangles and triangular traversal. |
| `scheduling.sparse` | CSC traversal, right-hand-side grouping and sparse accumulation schedules. |
| `kernels` | The `EngineKernels` arithmetic contract, `PanelWork` and shared kernel policies. |
| `kernels.scalar` | The complete scalar engine and numerical leaves over addressed windows. |
| `engine` | Engine selection, composition, provider inspection and whole-call routing. |
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
