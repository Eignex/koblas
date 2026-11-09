# Architecture

Koblas keeps public storage, operation contracts, exact engine access and vendor metadata in
`com.eignex.koblas`, and execution details under `internal`. All of these belong to the same
multiplatform module; platform source sets supply their own runtime and arithmetic implementations.

## Public entry points

- `Matrix` and `Vector` own fixed-shape mutable entry contracts and default materialization. `DenseMatrix`
  represents compact buffers and borrowed views with offsets and leading dimensions. `DenseVector`
  represents contiguous buffers and stepped or reversed slices with offsets and strides. Sparse setters
  maintain sorted structure and retain written zeros. Dense equality and serialization visit logical entries only. Engine eligibility checks layout rather than concrete types.
- `KoblasEngine` owns typed BLAS calls over dense and sparse containers. Logical dimensions are validated
  before views are unpacked. Its matrix methods prepare their operands and invoke shared schedules.
  Dense vector views survive validation, whole-call dispatch and alias preparation. Matrix product windows
  retain matrix views through schedule selection, then unpack for packed or panel arithmetic. Internal
  array-window adapters call the same typed methods; windowed `ger` keeps its direct panel schedule.
- Column views provide mutable vector access without copying matrix entries. Concrete containers retain
  typed dense/sparse callbacks; base `Matrix` helpers also adapt custom storage. Sparse vectors identify
  their packed window by offset and count. A sparse column view holds its owner and column index, so
  structural replacements update the matrix and every retained view resolves the current CSC window.
- `VectorOps.kt` and `MatrixOps.kt` are small container facades over the shared `koblas` variable in
  `KoblasEngine.kt`. Both facades share the `Norm` enum, with norm dispatch and validation under
  `internal/operations`. Every facade extension targets `Matrix` or `Vector`; copies and matrix
  transposes are storage transformations, and copy-into operations validate shapes and stage aliases.
  Sparse reductions traverse stored entries; custom storage uses scalar readers. Explicit BLAS calls,
  including vector scaling and scaled addition, stay on the engine.
- Applications may assign `koblas` during startup; launch settings initialize its first value. Callers own
  startup order and safe publication. Reads and writes have no lazy delegate or synchronization.
- `KoblasEngine.scalar`, `KoblasEngine.simd` and `KoblasEngine.vendor` expose exact engines;
  `Vendor`, `EngineImplementation` and `ThreadEvidence` describe their capabilities and provenance.

## Internal ownership

| Package | Owns |
|---|---|
| `storage` | Bounds, pooled arrays, physical windows and structural CSC utilities. |
| `operations` | Reusable BLAS validation, alias snapshots, scratch loans and vector storage adaptation. |
| `scheduling.dense` | Panels, packing, product blocking, selected triangles and triangular traversal. |
| `scheduling.sparse` | CSC traversal, right-hand-side grouping and sparse accumulation schedules. |
| `kernels` | Independent `EngineKernels` arithmetic and geometry, the `KernelProvider` engine bridge, and array-window adapters. |
| `kernels.scalar` | The complete scalar engine and numerical leaves over addressed windows. |
| `engine` | Startup selection, fixed kernel roles, one optional host library, host facts and measured cutoffs. |
| `host` | Shared host-library contracts and ABI adaptation. |

A default matrix method prepares its operands and calls a shared schedule, which invokes the selected
kernel provider. Preparation settles shapes, empty cases, BLAS no-read rules and aliasing before
arithmetic writes the destination. The built-in engine can hand an eligible whole call to its selected host library;
that binding owns its argument preparation and numerical library call.

Schedules own traversal and workspace extents. Kernels own arithmetic and recommend grouping or packing
geometry through their contract. The schedule can process a smaller group; group width, register tile,
cache block and diagonal block are independent quantities. Scalar and SIMD engines share the schedules.

`EngineKernels` has no public BLAS operations or dependency on engine selection and preparation.
`KernelProvider` combines that arithmetic with `KoblasEngine` for implementations that are also engines;
it adds no dispatch layer. Backend geometry and body attribution stay with the numerical callbacks they
describe. Scalar leaves depend on storage and kernel contracts. They do not invoke operation preparation
or shared schedules. Sparse Level 2 preparation and traversal live in ordinary operation functions and
shared CSC schedules, regardless of which Level 1 engine is selected.

## Platforms

JVM Vector API kernels remain under `jvmJava25/internal/jvm25/simd`. JVM and native host bindings remain
in their platform source sets under `internal/host`. Platform selection uses `internal/engine` actuals.
Klause and Kumulant consume the public engine and containers; platform implementation and build setup
stay within Koblas.

Tests mirror the internal package that owns the implementation. Public API and published-artifact
checks exercise calls through the engine and container entry points.

`BuiltinEngine` binds fixed vector, indexed, panel, product and triangular kernel roles, plus at most one
host library. Automatic selection stops at the first available compatible vendor. Missing exports and calls
below its cutoffs use the owned kernels. Internal factories keep kernel types through selection; tuning
binds the selected vendor once. Runtime dispatch and attribution share provider lookup.

Tuning cutoffs live in `internal/engine/EngineTuning.kt` as direct predicates; unmeasured targets borrow
compatible estimates. The benchmark fitter reports cutoffs for one backend and retains target provenance
for review before those predicates are updated.
