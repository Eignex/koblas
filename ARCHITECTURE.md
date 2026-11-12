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
  retain matrix views through schedule selection, then unpack for packed or panel arithmetic. The internal
  windowed `ger` adapter keeps its direct panel schedule.
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
| `storage` | Shared bounds, pooled arrays, serialization and storage transformations. |
| `storage.dense` / `storage.sparse` | Dense windows and copying / CSC column structure and transpose. |
| `operations` | BLAS window adapters, validation, alias staging and matrix/vector storage adaptation. |
| `scheduling` | Representation-independent panel iteration. |
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

Vector API values, masks and shuffles stay inside a concrete backend entry: project helpers
that carry them are Kotlin `inline` functions. Ordinary primitive/array kernel boundaries may
remain outlined. Blanket scalar inlining grows callers and can move compilation cliffs rather
than remove them.

Higher-order arithmetic helpers are source-inlined, or accept primitive callback interfaces.
The packaged boundary check rejects generic `FunctionN` arguments entering kernels, schedules
or operation preparation. Startup factories, buffer factories used only on pool misses and
compiler-generated public facade bridges may remain outlined.

Vector backend expression and cast null assertions have a local non-null fast path in the
packaged artifact. Only null reaches the original Kotlin failure helper; argument evaluation,
exception behavior and casts remain intact. Scalar code and parameter checks are unchanged.
`verifyKernelBoundaries`, part of `check`, rejects typed project calls carrying Vector values
and unguarded Vector backend assertions, including method-handle and lambda-bootstrap references.
It reads baseline and Java 25 bytecode independently of Kotlin metadata. These checks enforce
representation boundaries; throughput and allocation checks still validate compiled kernels.

## Platforms

JVM Vector API kernels remain under `jvmJava25/internal/jvm25/simd`. JVM and native host bindings remain
under `internal/jvm25/host` and `internal/host`. Platform selection uses `internal/engine` actuals.
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

`koblas-bench` owns CPU model and active SVE-width probes and formats thread evidence for reports.
The library detects only the host facts used by engine selection and tuning.
