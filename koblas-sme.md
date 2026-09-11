# SME, SME2, and extensible CPU kernel architecture

Status: proposed end-state architecture. This is an implementation plan, not a claim of measured speedups.
**Backward compatibility is not required.** Optimize for the final architecture and measured performance.
Existing public/internal Kotlin APIs, C ABI and symbols, packed formats, engine composition, configuration keys,
and tuning organization may be changed or removed. Existing consumers may need to rebuild and update their
calls; preserving source, binary, configuration, or packed-data compatibility is not an acceptance criterion.
Do not add deprecated aliases, legacy modes, dual APIs, or compatibility shims solely to preserve old usage.

Temporary internal adapters are allowed only to keep the numbered PR transition buildable while in-repository
consumers migrate. Each needs a removal owner and must disappear once its last consumer migrates. Update
callers, tests, API dumps, and documentation in the owning PR. Preserve the required mathematical, aliasing,
and no-read semantics; backward compatibility does not justify retaining old interfaces or constrain new ones.
Versioning and layout validation in the new design protect correctness and future evolution, not legacy support.

The objective is the best measured end-to-end implementation for each supported operation, shape, hardware,
and runtime. Build independently selectable SME and SME2 implementations. Design the same interfaces for future
AMD/Intel extensions, including AVX10, AMX, and the joint AI Compute Extensions (ACE). Choose among scalar Kotlin,
JVM Vector API, ordinary C SIMD, and eligible matrix accelerators at the appropriate operation boundary.
Keep mathematical semantics, explicit backend testing, reusable storage, and deterministic selection.

Keep the implementation proportional to demonstrated needs. Build the shared contracts and both SME backends;
add specialized layouts, schedules, or storage strategies when measurements justify their maintenance cost.
Sparse-dense analysis, new standalone streaming-vector kernels, new vendor bindings, and future low-precision
metadata are conditional follow-ups, not prerequisites for completing the dense SME/SME2 transition.

**1. Decisions that define the end state**

1. Use one versioned, architecture-neutral C probe for compiled kernels and usable hardware capabilities.
2. Separate hardware facts, kernel/layout descriptions, native scheduling recommendations, and JVM crossover
   policy. A feature bit alone never establishes the fastest implementation.
3. Replace engine-wide scalar/SIMD/C precedence with immutable, operation-specific execution plans.
4. Make a matrix block or panel the native execution unit. Microtiles stay inside the native implementation;
   the JVM does not enter C or streaming mode separately for every microtile.
5. Make packed data self-describing. Its logical dimensions, physical strides, layout version, and padding
   belong to the packed object, not to an implicit process-wide tile size.
6. Separate ordinary vector width, matrix register microtile geometry, packing geometry, cache blocking, and
   triangular diagonal-block size. They are independent tuning dimensions.
7. Support direct, one-operand-packed, both-operands-packed, and retained-packed matrix execution plans.
8. Fuse scaling and final writeback into matrix kernels where the numerical contract permits it. Avoid an
   unconditional separate beta pass, tile scratch copy, or alpha-scaled repacking.
9. Build both SME and SME2 paths, with exact selection and attribution. An SME2 machine must also be able to
   execute the SME-only implementation for comparison.
10. Determine defaults offline using the repository benchmark harness. No startup benchmarking, mutable
    process-global engine installation, or external provider discovery is needed.
11. Describe numerical types and accelerator state explicitly. A matrix extension is not just a wider SIMD
    vector, and a low-precision matrix unit is not eligible for an exact Double operation.

**2. Current code that motivates the redesign**

The following observations describe the current checkout, not constraints on the replacement:

| Current area | Limitation to remove |
|---|---|
| `KoblasEngine.kt`, `BuiltinEngines.jvm.kt` | Default selection chooses the whole Vector API engine ahead of the whole C engine. The engine name does not identify packed implementations. |
| `PackedKernels.kt`, `CPackedKernels.kt`, C header | C uses a 4 by 4 physical tile. The interface binds packing, GEMM, and triangular solve geometry together. |
| `SimdGemmTile.kt` | Its physical shape is two preferred JVM vectors by four columns. That shape is unrelated to an SME tile. |
| `PackedGemm.kt`, `PackedTileOutput.kt` | Kotlin walks every microtile; edges and diagonal tiles use scratch and copyback. |
| `PackedLayout.kt`, `PackedPanels.kt` | Packing uses raw arrays and engine-implied shapes; public and internal packing have no accelerated layout backend. |
| `DenseGemvKernels.kt`, `DensePanelKernels.kt` | GEMV is expressed as repeated `axpy4` or `dot4`, with output traffic and call boundaries every four columns. |
| `DenseSymvKernels.kt` | Group size four and its numerical checks are embedded in the portable traversal. |
| `PackedTriangularSolve.kt` | Diagonal solve order follows packed tile columns; many row tiles are called separately. |
| `DenseBlas.kt` view overloads | Some strided/view paths execute separate portable implementations and bypass engine acceleration. |
| `DenseTuning.kt`, `SparseTuning.kt` | Process-wide constants mix cache blocks, native choices, and host-call thresholds across hardware. |
| `CVectorKernels.kt` | Several ordinary mutations always remain in Kotlin based on earlier measurements; new native widths need fresh comparisons. |
| `koblas_kernels.def`, build script | Implementation is compiled through the cinterop header, while JVM builds a shared library. ISA variants need a common separately compiled source tree. |

**3. C probe and kernel catalog**

Expose an ordinary, non-streaming C ABI. The probe must work on baseline hardware, even when the same library
contains instructions that hardware cannot execute.

Proposed entry point:

```c
int32_t koblas_probe_v1(
    const koblas_probe_request_v1 *request,
    koblas_probe_result_v1 *result);
```

Use three query kinds: `HOST`, `KERNEL`, and `THREAD`. `KERNEL` can enumerate built implementations or resolve
an exact kernel ID. All structs begin with ABI version and byte size; use fixed-width integer fields, explicit
padding, documented alignment, and reserved zero fields. Return explicit status codes. Never export C++ types,
scalable-vector types, compiler-specific enums, or compiler vector structs through the ABI.

Use extensible feature IDs/records and versioned structs, without building a general capability-query framework.
Start with host/context inspection, catalog enumeration, and exact ID lookup. Filter candidates in the shared
planner; add query-side filters only if catalog size makes them necessary. Do not reduce eligibility to `has_matrix`.
Use operation-specific type tuples: A type, B type, accumulation type, output type, and numerical mode. Future
typed wrappers can reuse the catalog without adding dtype branches to a Double hot loop.

| Query/result | Required information |
|---|---|
| Host | Architecture, OS, compiled feature set, OS-usable feature set, catalog revision, conservative CPU tuning key, optional CPU-model evidence. Unknown model is valid. |
| Kernel request | Enumeration cursor or exact implementation ID. |
| Kernel result | Stable implementation ID, operation/type tuple, required features, availability/reason, supported semantics/addressing modes, execution-state requirements, width/geometry, layout IDs, and scratch/alignment constraints. |
| Thread | Current ordinary SVE VL and SME SVL when supported, relevant execution-state readiness, plus validity flags. Unsupported and unknown are distinguishable from zero. |

Operation families include vector reductions, vector mutations, generalized dot/update panels, GEMV blocks,
SYMV blocks, rank updates, GEMM blocks, triangular solve/update blocks, packing, unpacking, and transpose.
The catalog contains only implementations built into Koblas. It is an immutable table, not a registration API.

The probe reports available alternatives and their legality. The shared Kotlin planner owns performance choice,
using runtime-specific profiles; avoid a second recommendation policy in C. An exact request never silently
resolves to another ISA or width. AUTO may choose a documented fallback.

Typed execution entry points take the resolved kernel ID and ordinary array pointers/offsets/strides. For
example, `koblas_gemm_block_v1(kernel_id, ...)` and `koblas_dot_v1(kernel_id, ...)`. C switches or indexes a
validated static table once at entry, never inside the arithmetic loop. Keep IDs private to the ABI contract;
never persist a raw function address or rely on catalog ordinal stability. A stale/incompatible descriptor is
rejected before mutation. Baseline wrappers perform ISA eligibility checks; hot kernels assume validated input.

Report actual execution variants in diagnostic/benchmark mode, including any AUTO fallback. A kernel family
whose GEMM uses SME but whose diagonal solve uses ordinary SIMD must describe those components accurately.

Capability rules:

- Linux: use OS-advertised capabilities for SME, SME2, and optional FP64 support. CPU ID registers alone do not
  establish usable OS support. Ordinary SVE support and SME support are distinct.
- macOS: use exact feature queries for SME, SME2, and SME FP64; missing/unreadable queries mean unavailable to
  that selection path. Avoid chip-name-only feature inference.
- x86: the future-width mechanism must account for OS register-state support as well as CPU ISA support.
- Probe compiled support independently from hardware support. A capable host with an older compiler build
  must receive a truthful unavailable result.

Linux tracks streaming state and SVL per thread. Never treat the initializer thread's vector length as a global
packed-layout guarantee. [Linux SME interface](https://docs.kernel.org/arch/arm64/sme.html)

**4. Ordinary one-dimensional widths and SME vector lengths**

Each kernel descriptor distinguishes:

- execution ISA and execution mode: scalar, ordinary SIMD/SVE, or streaming;
- fixed register width, or vector-length-agnostic width mode;
- logical elements processed per iteration;
- unroll factor and number of independent accumulators;
- matrix microtile and packed-layout geometry, where relevant.

Accelerator descriptors identify primitive kind, legal M/N/K granularities, state requirements, and supported
tail/layout modes. Keep register allocation, transfer schedules, and other details inside the native kernel
unless a planner actually needs them. Do not represent AMX or ACE with an invented SME streaming-vector length.

Do not encode all of this as `lanes`. The current C `vector_size(32)` source type can be split into narrower
machine instructions. A four-column panel is also not a four-lane vector.

Build the framework with baseline C and existing ordinary SIMD variants. Instantiate ordinary fixed-width
leaves through shared templates/specializations with explicit backend IDs. Allow later NEON, SSE2, AVX2,
AVX-512, and SVE variants without changing the probe or Kotlin execution interfaces. Only advertise variants
actually built and verified. Wider ISA support does not automatically make that variant preferable.

SME/SME2 kernels read SVL within the correctly attributed execution context. Their arithmetic strip-mines logical
blocks according to that SVL. Packed physical strides are explicit and remain interpretable when a retained
panel is used on a different thread. Native microtiles can use predicates and multiple ZA tiles; a physical
packing block is not required to equal one hardware tile.

For a specialization requiring a particular SVL, check the requirement before touching output. AUTO either uses
a compatible implementation for that layout or repacks through explicit scratch. Exact execution reports a
mismatch. Never change the caller's SVL as an implicit optimization. Querying a thread may inform a plan, but
the kernel entry remains responsible for its own execution preconditions.

**5. Kotlin execution architecture and public low-level API**

Use these responsibilities; they need not each become a new public type, interface, or registry. Keep internal
contracts internal unless an existing custom-algorithm or retained-operand use case needs public access:

| Component | Responsibility |
|---|---|
| `KernelCatalog` | Immutable decoded C descriptors and JVM/scalar implementation descriptions. |
| `KernelProfile` | Resolved hardware/runtime tuning and supported candidate choices. |
| `DenseExecutionPlan` | Selected operation algorithm, kernel IDs, layout, cache blocks, scratch, and host-call strategy. |
| `VectorKernels` | Length/stride-based one-dimensional execution; internal machine width is invisible to callers. |
| `PanelKernels` | Variable-width multi-dot, multi-column update, and coupled symmetric updates. |
| `MatrixKernels` | Logical matrix-block products, symmetric products, triangular solves/updates, and rank updates. |
| `LayoutKernels` | Packing, unpacking, and transpose over explicit windows and structural flags. |
| `PackedLayout` and packed operands | Layout version, role, logical dimensions, strides, physical padding/alignment, buffer and offset, and any baked transform/scaling. |

The global engine is an immutable AUTO policy, selected once. Its per-call choices depend on operation metadata
and current execution constraints; this does not require mutable global providers. It can use JVM SIMD for a
vector operation and SME2 for GEMM within the same engine.

Keep one portable orchestration per Level 2/3 operation, with generic direct/packed/ordered strategies and an
independently callable scalar oracle. Do not duplicate portable algorithms into SME and non-SME families.

Expose explicit engine/kernel selection for scalar Kotlin, JVM SIMD, ordinary C, SME, and SME2, with exact kernel
IDs/widths available to tests and the benchmark harness. Existing `BuiltinEngines` and `PackedKernels` APIs can
be replaced; do not retain misleading compatibility shims. A composed experimental engine may use baseline
components, but its name and per-operation descriptors must reveal them. Exact benchmark mode fails when the
requested accelerated implementation cannot run.

Replace untyped retained packed arrays with layout-bearing operands. Advanced callers may wrap a caller-owned
buffer with a validated descriptor. Compute validates layout compatibility; it cannot reinterpret a 4-by-4
packing as a 16-by-16 packing. Preserve a low-level exact block execution API for custom algorithms and tests,
but do not require every backend to expose the same microtile dimensions.

Unify owning matrices and views through a matrix-window description: backing array, offset, logical dimensions,
row/column strides, transpose, and structural flags. Kotlin handles validation and permitted alias staging once;
selected kernels operate on the validated windows. Update the current view documentation to distinguish normal
internal panel packing from materializing a whole contiguous matrix. Support zero-copy direct execution when
its addressing requirements hold.

A representative operation follows:

```text
validate shape and alias rules
resolve semantic eligibility and zero-work behavior
choose direct / packed / retained-packed algorithm and implementation
allocate or borrow the selected scratch once
execute bounded panel/block calls with scaling and writeback
return with no native state or borrowed addresses retained
```

Separate plan selection from execution so tests can inspect a decision without performing arithmetic. Avoid
allocating a plan object on every small call: small immutable plan variants and primitive dispatch fields can
be prebound; only dimension-dependent scratch sizes and offsets need calculating per operation.

**6. Kernel inventory: what gets SME and SME2 implementations**

SME FP64 outer products and SME2 FP64 vector-group arithmetic require the optional FP64 feature. SME2 adds
grouped arithmetic/transfers; it does not replace the need for the SME outer-product primitive in a GEMM
implementation. Keep the two ISA implementations separately compiled and attributable.
[Arm ACLE](https://arm-software.github.io/acle/main/acle.html)

| Existing operation or kernel | SME implementation | SME2 implementation / final interface |
|---|---|---|
| `gemmTile` | FP64 `FMOPA`, multiple accumulators/ZA tiles, general edge handling. | Distinct implementation using `FMOPA` plus grouped movement and optimized writeback. Both sit inside a logical GEMM-block API. |
| `gemmTrsmTile` | Product subtraction followed by substitution in the same native call. Keep residuals in registers where useful. | Grouped residual transfers and updates across RHS vectors; compare against the SME schedule. Replace with update-and-solve block execution. |
| `trsmTile` | Streaming-vector substitution across independent RHS rows, plus an ordinary SIMD competitor. | Grouped updates across independent RHS rows. Division and pivot order remain dependency-bound; do not pretend the solve itself is a GEMM. |
| `packLeftLayout`, `packRightLayout` | Panel loads and ZA-assisted rearrangement where profitable. | Grouped transfers/transposition, suitable general/symmetric/triangular packers. Fuse optional alpha only when the plan explicitly requests it. |
| `writeLeftLayout`, `writeRightLayout` | Vectorized unpack/writeback for layout-bearing panels. | Grouped extraction and transpose where profitable. |
| `transposeBlocked` | ZA horizontal/vertical traversal over logical blocks. | Grouped ZA transfers and tuned memory scheduling. |
| `dot4` | General multi-dot panel reduction; retain ordinary SIMD competition. | Variable-width multi-dot using grouped accumulation. No API limit of four outputs. |
| `axpy4` | General column-panel update retaining output across more columns. | Grouped updates with output vectors retained in ZA across the panel. Preserve required coefficient evaluation. |
| `dotAxpy` | Coupled dot/update leaf or symmetric block implementation. | Coupled grouped accumulation; prefer a whole SYMV block when that avoids repeated transfers. |
| `axpyArithmetic` | Ordinary/streaming vector alternative inside larger matrix calls. | Grouped matrix-update alternative; preserve evaluation of zero multipliers. |
| `ger`, `syr`, `syr2` update loops | Direct outer-product tile updates with structural and zero-coefficient masks. | Grouped load/store and update variants. Keep a bandwidth-efficient ordinary SIMD alternative. |
| `clearLeftPadding`, `clearRightPadding` | Ordinary fill remains a valid best implementation. | No dedicated streaming region solely to clear a short padding run. Fuse clearing into packing where possible. |

The end-state catalog includes alternative schedules for applicable operations. AUTO only enables those that win
representative benchmarks. Having an SME2 backend does not mean forcing every operation through SME2.
Implement logical modes through shared packing/scheduling where possible; do not require a distinct native
kernel for every ISA, layout, transpose, and tile-size combination. Add a specialization only for a measured gap.

Standalone `dot`, `sum`, `ssqd`, `asum`, `axpy`, `scale`, `swap`, and rotations participate in the width-aware
catalog. Evaluate SME2 reductions and grouped arithmetic, but retain ordinary SIMD for memory-bound cases.
`nrm2` must keep robust rescaling, NaN/Inf behavior, and overflow/underflow handling; a raw ZA sum of squares is
not an acceptable replacement. `rotmg`, irregular indexed operations, and dependency-heavy sparse work need
no artificial SME implementation to make the engine appear uniform.

**7. Required changes to portable Level 2 and Level 3 algorithms**

This is an algorithm/interface redesign, not just substitution of C leaves.

| Portable area | End-state change |
|---|---|
| `denseGemvUpdate` | Replace the fixed-four-column traversal with a GEMV planner and variable-width block calls. For non-transpose, hold a y block across a column panel. For transpose, hold multiple reductions across a row panel. Pass offsets/strides directly. |
| `symvUpdate` | Replace hard-coded four-column regrouping with symmetric blocks. An off-diagonal block contributes both `Ablock*x` and its transpose contribution; diagonal blocks read only the stored triangle. Fuse both contributions where useful. |
| `packedProduct` / `macroKernel` | Replace Kotlin microtile iteration with cache-block scheduling and one native call per suitably bounded block. A native loop owns microtiles and streaming lifetime. |
| GEMM entry paths | Select direct, one-side-packed, both-sides-packed, or retained-packed execution before preparing data. Treat skinny shapes and small depth as distinct categories. |
| `symm`, `gemmt`, `syrk`, `syr2k` | Share the new product infrastructure with explicit structural packing/output masks. Add a fused rank-2k update so both products can share output traffic; retain the reference evaluation path when needed. |
| `accumulatePackedProductTile` | Move edges, selected-triangle stores, alpha/beta, and final writeback into block execution. Native masked stores replace routine Kotlin scratch/copyback. |
| `packedTrsmCore` | Use independently tuned diagonal blocks and RHS panels. Separate dependency order from GEMM packing widths; update many RHS tiles per native call. Fuse the final update with the diagonal solve when beneficial. |
| `packedTrmmCore` | Use block products with correct dependency traversal or a source snapshot chosen by the plan. Share source packing and output handling across microtiles. |
| `gerUpdate`, `syrUpdate`, `syr2Update` | Add direct rank-update block calls instead of one AXPY call per matrix column. Preserve selected-triangle and zero-source behavior. |
| Strided/view entry paths | Route through the same planner rather than separate scalar default bodies. General strides remain supported by direct kernels or explicit panel packing. |
| Packing/transpose paths | Route through `LayoutKernels`; portable code describes windows and structure, native code performs the chosen rearrangement. |

Keep a separately callable scalar reference implementation for every new operation contract. Existing ordered
fallbacks and numerical guards are valuable evidence, but they can be refactored into explicit semantic
eligibility checks instead of being entangled with a specific four-column traversal.

SYMV regrouping can change which intermediate overflows even when inputs are finite. Assess the new block
schedule against the existing checks. A fast-path refusal must occur before externally visible writes; otherwise
compute in scratch and commit only after the relevant check. Never retry the full reference operation on a
partially updated destination.

TRSV/TRMV retain dependency-aware substitution and vector/panel updates. Ordinary SIMD is a serious candidate
for a single RHS. Do not expand them into dense GEMM or introduce extra asymptotic work to occupy ZA.

Migrate existing sparse consumers of changed panel APIs without changing their algorithms. A new sparse-dense
panel strategy is a follow-up only after packing/gather-inclusive measurements demonstrate enough RHS reuse or
subblock density. Do not require new prepared sparse analysis or invalidation machinery for the dense rollout.
Generic sparse-sparse, indexing, and single-vector algorithms remain unchanged. HFactor integration is separate.

**8. Packed layouts, tile shapes, and scaling**

Use a small catalog of documented formats rather than one universal format:

- A general packed left/right format with explicit depth stride and row/column groups, usable by ordinary SIMD
  and SME implementations that understand that layout.
- Backend-specialized formats when measured packing/compute gains justify them. Compatibility is keyed by
  layout version and geometry, not merely by ISA name.
- Explicit retained panels, preferably unscaled, so changes in alpha do not require rebuilding them.

Descriptors identify any alpha or transpose baked into packing. The planner avoids applying them twice. A
kernel consumes a declared logical window and only touches its permitted physical input/output windows.
Predication handles tails; padding is a format requirement when explicitly declared, not an accidental universal
requirement inherited from the old API. Unused triangle data and unit diagonals must not be loaded.

Tune these independently:

| Quantity | Meaning |
|---|---|
| `mr`, `nr` | Output microtile geometry inside one native schedule. |
| packing row/column group and alignment | Physical reusable operand format. |
| `mc`, `nc`, `kc` | Cache-resident block dimensions. |
| `rhs_block`, `diag_block` | Triangular scheduling dimensions. |
| GEMV/SYMV row and reduction blocks | Retention/reuse of vector outputs and matrix data. |
| native-call work limit | Bound on each JVM critical-call batch. |
| reduction accumulators/unroll | Latency hiding for ordinary and streaming vector operations. |

At a 512-bit SVL, investigate 8-by-8, 16-by-8, 8-by-16, and 16-by-16 FP64 GEMM microtiles, then larger logical
blocks assembled from them. These are search candidates, not promised optimal defaults. Include narrow tiles
and other SVLs. Do not assume one cache block or TRSM block is optimal just because it matches an FMOPA tile.

For a depth-blocked product, beta is applied on the first contribution and subsequent contributions accumulate.
For a fused rank-2k operation, apply beta once across both products. beta zero must avoid reading original C;
alpha zero and empty dimensions retain the operation-specific no-read behavior. An unscaled packed format and
epilogue alpha are an algorithm choice, not permission to change overflow/zero semantics silently.

**9. Tuning data and default settings**

Replace global tuning objects with resolved typed profile data under `koblas/src/tuning/`. Start with compact
checked-in tables and a small deterministic validator/importer for benchmark-derived settings. Kotlin owns
selection and passes the selected schedule/work limits to C; legal native variants remain in the native catalog.
Add cross-language generation only for fields that actually need to be consumed in both languages. Do not build
a configuration language, duplicate performance policy in C, or require runtime JSON/config downloads.

A profile key includes architecture/CPU family when reliably identified, native implementation ID and layout,
runtime (`native`, `jvm-c`, `jvm-auto`), and JVM preferred vector width when relevant. Record the JDK/compiler
versions used for calibration; use a conservative profile when calibration does not cover the running system.
Do not generate a new profile for every benchmark machine or every patch-level runtime version automatically.

Separate the data into four layers:

1. **Structural catalog:** supported instructions, formats, legal geometries, alignment, and scratch formulas.
   Tuning cannot make an unsupported kernel legal.
2. **Native schedule:** preferred implementation/width, microtile shape, cache blocks, unroll, prefetch choices,
   and triangular/RHS blocks for each operation category.
3. **Host-call policy:** scalar-to-C, JVM-SIMD-to-C, and Kotlin/Native-to-C decisions, with independent thresholds.
4. **Algorithm policy:** direct versus packed, one versus two packed operands, rank-update versus product,
   retained-data reuse, and batch size limits.

Specific measured CPU profiles override generic ISA/runtime profiles, which override portable defaults.
User overrides resolve once, with JVM properties ahead of environment variables. Expose only useful tuning keys;
remove obsolete ambiguous keys as part of the API/configuration migration. Resolve each valid override through
the schema before freezing the profile. Reject impossible geometry, negative work, overflowing scratch sizes,
or unsupported IDs. Malformed performance overrides fall back with an inspectable diagnostic; an explicit
unavailable backend request fails rather than silently changing its meaning.

Represent thresholds as `Never`, `AlwaysEligible`, or a typed minimum/shape rule. Avoid using `Int.MAX_VALUE`
as an infinity sentinel or forbidding zero when a benchmark needs to force eligibility. Use checked or
saturating 64-bit work estimates; `m*n*k` must not overflow an Int during dispatch.

Every tuned entry needs provenance: workload/fixture version, hardware evidence, compiler/JDK, native ID,
layout, comparison paths, report location, and the measured range where the choice holds. Remove obsolete
performance comments tied to a source file that now dispatches among several architectures.

There is no defensible universal numeric SME threshold yet. Calibrate the new defaults as part of completion.
Existing 4-by-4 depth thresholds and old x86 scalar-to-C measurements must not be copied into SME profiles.
Unmeasured kernels remain available in exact mode; AUTO uses the proven conservative alternative until the
profile has evidence. The deliverable includes measured SME and SME2 profiles for the available target hardware.

**10. JVM scalar, SIMD, and C dispatch**

Yes: the final AUTO engine needs JVM-SIMD-to-C thresholds on SME-capable hardware, separately from JVM-scalar-
to-C thresholds. It also needs the same mechanism for future ordinary C widths.

| Dispatch decision | Inputs and calibration |
|---|---|
| Scalar Kotlin versus JVM SIMD | Operation, length/stride, JVM vector species and warm JIT behavior. |
| Scalar Kotlin versus ordinary C/SME/SME2 | Full foreign-call and array-segment cost against actual JIT-compiled Kotlin, which may itself autovectorize. |
| JVM SIMD versus ordinary C/SME/SME2 | Separate measurements; the scalar-to-C threshold is not transferable. |
| Ordinary C versus SME versus SME2 | Native work and data-movement cost for compatible algorithms/layouts, including streaming transitions. |
| Direct versus packed product | `m`, `n`, `k`, transposes/strides, structure, packing bytes, and retained-operand reuse. |
| Kernel/block call versus larger batch | End-to-end throughput and safepoint latency, not only fewer foreign calls. |

Use small measured decision tables by operation and shape category; introduce a fitted cost model only if it
predicts held-out cases better than simple rules. For vectors, length and stride often suffice. For GEMM, a
single element or flop threshold does not: 1-by-N, small-k, and square products have different packing costs.
Do not require a scalar -> SIMD -> C ordering if measurements skip one of those stages.

Choose algorithm and layout before packing. Once a retained panel is provided, restrict candidate kernels to
compatible formats or explicitly cost a repack. For already packed block calls, calibrate thresholds for that
layout and edge shape; they do not include the original packing cost.

Keep scalar/SIMD fallback inside the JVM when selected. Do not cross into C merely for a probe that discovers
the operation is too small, and do not pay a second hidden scalar crossover after selecting an exact native
kernel. Cheap metadata decisions use cached capability/profile data; necessary thread-length checks occur
only on native-eligible paths and are rechecked by relevant native wrappers.

Scalar Kotlin, pure JVM SIMD, ordinary C, SME, SME2, and AUTO are separate benchmark identities. Raw kernel
benchmarks bypass performance thresholds but still honor zero-work semantics and safety preconditions. Normal
`jvm-c` policy can retain a Kotlin-small/C-large split; it must be distinguished from a forced raw C run.

**11. Native build and JVM memory/call boundaries**

Move implementations out of the cinterop header. Use declaration-only public ABI headers, internal headers,
baseline dispatch sources, ordinary SIMD sources, SME sources, and SME2 sources. Suggested native organization:

```text
src/nativeInterop/kernels/
  include/koblas_probe.h, koblas_vector.h, koblas_matrix.h, koblas_layout.h
  dispatch/probe.c, catalog.c, entrypoints.c
  scalar/...
  simd/...
  arm/sme/...
  arm/sme2/...
  internal/...
```

One build pipeline emits the JVM shared library and Kotlin/Native static archives from the same source set.
Kotlin/Native cinterop consumes declarations and links/embeds the correct target archive through Gradle. Handle
host builds and cross targets explicitly; never accidentally link the host archive into Linux arm64 or macOS
artifacts. Update resource checks, exported-symbol checks, task dependencies, and incremental build inputs for
compiler version, target triple, flags, generated profiles, and all implementation files.

Compile each ISA implementation separately. Baseline code must not acquire SME instructions through global
target flags, automatic vectorization, or LTO leakage. The SME-only object must not acquire SME2 instructions.
Verify this by disassembly and by executing with the newer feature unavailable. Build capability is determined
by compile/link probes, not solely by a compiler version comparison. Release builds intended to include SME
must fail if that requested component cannot be built; an explicitly baseline-only build remains supported.

Use compiler-managed streaming/ZA attributes behind the ordinary ABI, with no Kotlin callbacks inside streaming
regions. Validate platform ABI helper availability, prologue/epilogue preservation, and target-specific code
generation. A source-level ACLE compile success is not a completed link/runtime test.
[LLVM SME ABI lowering](https://llvm.org/docs/AArch64SME.html)

For JVM heap arrays, retain zero-copy critical FFM only for bounded compute calls. Batch a useful number of
tiles inside each call while enforcing profile-defined work/latency limits. Split very large reductions and
matrix work without breaking accumulation semantics. Measure GC/safepoint responsiveness as well as arithmetic
throughput. Add non-critical native-memory workspaces only if measurements show bounded heap-array calls cannot meet
throughput/latency needs. Include staging cost and ownership in that comparison; a second storage strategy is
not mandatory when the simpler path wins.

Probe calls that query the OS use ordinary non-critical FFM and small native result buffers. Kernel code keeps
no heap pointer beyond a call. Pin each required Kotlin/Native array once per block/panel call. Store scratch
in the caller's workspace, not in mutable global storage. Reusable packed/native buffers carry ownership and
lifetime rules, and concurrent callers require distinct mutable workspaces.

**12. Numerical and behavioral requirements**

Define the common contracts before tuning or broad rewrites:

- Scalar reference agreement within the existing appropriate rounding tolerance, with explicit classification
  checks for NaNs and infinities and bit checks where signed zero/padding/untouched storage is contractual.
- Robust norms, selected triangles, unit diagonals that are not read, zero work, and alpha/beta no-read rules.
- Distinguish standalone AXPY's zero-alpha shortcut from matrix arithmetic that evaluates zero multipliers.
- Preserve triangular zero-pivot/zero-source behavior. Do not replace division by an approximate reciprocal
  without a separately justified and tested contract.
- Bounds apply to actual backing buffers and logical windows, including offsets, leading dimensions, tails,
  padding, and allowed aliases. Wider unpredicated loads cannot read beyond a valid allocation.
- Do not use global fast-math flags to make the new implementation easier to vectorize. Audit contraction,
  reassociation, scaling placement, subnormals, overflow, and underflow for each changed schedule.
- A declined fast path must leave output untouched. Once mutation starts, complete that selected algorithm;
  failure recovery must not double-apply an update.

Refactor the scalar reference to express the new contracts, but do not define the oracle by calling the new C
implementation. Keep ordered reference paths for cases that cannot safely use a regrouped schedule.

**13. Validation and tuning workload**

Extend `koblas-bench`, whose README and compatibility rules remain the basis for measurements. Add exact ISA,
width, layout, and AUTO identities, with native probes recorded in reports. Current packed cases only cover
selected 4-by-4/8-by-4 physical shapes and skip mismatches; replace accidental coverage gaps with explicit
logical-shape and layout coverage for the new catalog.

**Tile and packed benchmark options — before baseline capture**

Make `koblas-bench/cases.txt` the source of truth for tile and packed workload settings in PR 01, before replacing
kernels. Each case must explicitly declare its applicable physical tile, versioned packing formats, packing
groups/strides/padding/alignment requirements, block/panel sizes, and timing mode. Missing required settings are
errors, including settings whose value happens to equal today's default. Do not reconstruct them from engine
tile properties, vector species, tuning profiles, or changing helper defaults. Both Kotlin and vendor runners
consume the same case specification; implementation code executes and validates it.

Keep mathematical case identity separate from implementation symbols. The runner may select the backend under
test, but cannot change the case's declared work. Parameter spelling must fit the existing `+key=value` schema.

| Control | Purpose |
|---|---|
| Logical dimensions and operation flags | Keep m/n/k, triangular solve order, transpose/side/uplo/diag, scaling, offsets, and strides equivalent across implementations. |
| Kernel/ISA and tile geometry | The case fixes physical microtile shape and any required variant constraints. Record the actual implementation ID; never use a backend default to supply missing geometry. |
| Packing configuration | Declare left/right layout IDs/versions, packing groups, applicable physical strides, padding, and alignment requirements in the case. Verify actual buffers match. |
| Block and panel schedule | Select supported cache blocks, panel width, diagonal/RHS blocks, and call batch size where applicable. Reject inapplicable or unsupported overrides rather than ignoring them. |
| Timing boundary | Separate raw tile, logical block, packing-only, prepacked compute, and pack-plus-compute. Record reset, allocation, staging, and retained-data reuse consistently. |

Provide two explicit comparison modes. Fixed-configuration comparisons require matching layout, tile geometry,
physical work, and timing semantics; they isolate kernel changes. Logical-workload comparisons allow different
layouts/microtiles for the same mathematical operation and timing boundary; they measure the complete chosen
strategy and report those differences. An old backend may cover a logical block with multiple old tile calls,
but their loop/call costs stay in the timed block. Never equate one larger tile call with a smaller tile's work.

Generate logical fixtures first, then pack them independently into each selected format. Physical padding must
not change logical inputs. Record logical work, physical extents/bytes, layout versions, actual kernels, and
source commit. Physical sizes may be calculated from the explicit fields and documented format formulas, but
not from hidden tuning settings. A layout ID's meaning is immutable; changing it requires a new version/case.
Keep AUTO/default-policy experiments explicitly labeled and separate from these fixed before/after cases.
Explicit unsupported selections produce a clear failure/unsupported row under the harness's existing policy;
no substitution and no arbitrary tile-size support fabricated by an option parser.

Keep old operation names or renamed successors associated through a versioned mathematical-work identity.
Do not join reports on a kernel symbol or remove physical metadata to force a match. Historical reports are
reusable only when their fixture and timing metadata proves equivalence; otherwise capture new baselines from
the old implementation with the extended harness. Version changed cases and update both Kotlin/vendor runners
and the comparator together. This requires no production compatibility API or permanent old-kernel copy.

Required tests:

1. Probe ABI sizing/version negotiation, unavailable features, exact-selection failures, ordinary width
   resolution, and deterministic generic/CPU-profile fallback. Test injected capability records without
   executing unsupported instructions.
2. Conformance for each new vector/panel/matrix/layout implementation against the explicit scalar oracle.
3. Full/partial tiles, depth tails, nonzero offsets, unusual leading dimensions and strides, triangle
   boundaries, unit diagonals, zeros, non-finites, extreme magnitudes, and backing-buffer aliases.
4. Retained panels used by compatible different kernels and by threads with different permitted vector
   lengths; incompatible exact calls fail before mutation.
5. All transpose/side/uplo/diag combinations for products and triangular operations. Perturb identity-pivot
   fixtures when checking permutation behavior.
6. JVM module absent, native access absent, C library absent/old/incomplete, SME without FP64, SME without
   SME2, and both available. Native target builds and links must be tested separately from JVM FFM.
7. Warmed allocation checks, thread safety, concurrent calls, native sanitizers/guarded-buffer checks, ABI
   state preservation, and call-boundary latency.

Required performance suites:

- Raw vector and panel kernels; prepacked blocks; packing/unpacking/transpose alone; full operations including
  scaling, aliases/scratch where relevant, packing, and FFM; repeated operations using retained panels.
- Tiny through cache-exceeding vector lengths, contiguous and strided storage, hot/cold data, and sweeps around
  each prospective threshold.
- Square, tall, wide, skinny, tiny-k and large-k GEMMs; dimensions around microtile/cache-block boundaries.
- GEMV both transposes, SYMV both triangles, rank updates, GEMMT/SYRK/SYR2K/SYMM, and triangular shapes spanning
  single-RHS to many-RHS cases.
- Separate ordinary SIMD, SME, and SME2 on the same hardware. Test SME-only instruction eligibility through
  suitable feature-restricted execution where available, then use real hardware for performance claims.
- Measure both throughput and latency for native-call batching, plus representative concurrent workloads on
  heterogeneous Apple CPUs. Do not derive all-core scaling by multiplying a single-core instruction result.
- Add Apple Accelerate as a benchmark-only vendor reference for macOS alongside the existing supported vendor
  comparisons if existing vendor coverage cannot answer a target comparison. This binding is an optional
  follow-up, not a release gate. Keep thread count, inputs, and timing boundaries explicit and compatible.

Generate tuning candidates offline, validate on held-out shapes and repeated runs, and prefer the simpler
choice when results overlap. Store reports and profile provenance. A raw kernel win with a full-operation
regression does not justify an AUTO default. Do not promise percentage gains before those runs.

Implementation verification uses `./gradlew :koblas:check :koblas-bench:check lintDocs` and
`./gradlew :koblas:check :koblas-bench:check -Pkoblas.noSimd=true`, plus native conformance and exact ISA runs. Use the repository's full check before pushing, and
HFactor-specific validation only if that module changes. No `.github/` edits are required by this plan.

**14. Implementation sequence and completion**

[koblas-sme-steps.md](koblas-sme-steps.md) is the sole PR sequence and verification-gate checklist. Each PR
starts in a fresh implementation session with an explicit session goal and its specified model/effort. The
session implements, commits, pushes, and OPENS a GitHub PR, resolves independent review findings, and gets
required CI green on the final reviewed head before marking its goal complete. An open PR with pending checks
is unfinished. Every step ends with independent review in a separate fresh context; merging is not automatic.
Do not maintain a second dependency roadmap here.

Completion requires both SME and SME2 dense backends, the generic probe and future-width seam, shared portable
orchestration, measured profiles, and verified numerical/storage contracts. Optional investigations may conclude
with evidence that the existing route wins; they do not require shipping unused kernels or speculative APIs.
No old interface, group size, or crossover survives solely for compatibility.

**15. Evidence and remaining empirical questions**

The earlier exploration compiled FP64 SME `FMOPA` and SME2 vector-group FMLA/ZA-transfer probes with locally
available Clang 19.1.4. Ordinary pointer-based entry points compiled for generic AArch64 and Apple M4. This
establishes source/code-generation feasibility only; complete link, ABI, numerical, and performance validation
is still required.

Published M4 work supports investigating both outer-product and vector-group paths, but does not supply
Koblas's thresholds or prove which packed format is best. Hardware measurements must resolve microtiles,
cache blocks, direct/packed crossover, SME-versus-SME2 schedules, and JVM call limits.
[Hello SME research](https://arxiv.org/html/2409.18779v1)

An implementation is complete when the best-supported choice is selected for measured workloads, slower
alternatives remain explicitly testable, and unsupported/unmeasured systems take a correct, explainable path.
The target is an extensible performance architecture, not an assumption that the newest ISA always wins.

**16. Future AMD/Intel extensions: ACE, AMX, and AVX10**

ACE means AI Compute Extensions in the joint AMD/Intel publication. The published v1.15 specification describes
AVX10 input vectors, tile accumulators, outer-product instructions, tile/vector transfers, and block-scale state.
Its initial accumulator types are FP32 and INT32; it does not specify a Double-precision matrix arithmetic path.
Do not advertise ACE v1 as a DGEMM backend or silently down-convert Double inputs. Treat specification support,
compiler support, OS support, and available hardware as separate facts.
[ACE specification v1.15, sections 2, 10, and 15](https://x86ecosystem.org/wp-content/uploads/2026/06/ACE_v1_Specification_public_1_15.pdf)

The shared architectural pattern of vector inputs, outer-product accumulation, and vector-assisted final
processing makes ACE a useful design test for the proposed block interface. Its packing and state handling
still belong to its own backend. [Joint ACE whitepaper](https://x86ecosystem.org/wp-content/uploads/2026/03/ACE-Whitepaper-v1.pdf)

| Extension family | Immediate architectural provision | Eligibility for today's Double API |
|---|---|---|
| AVX2 / AVX-512 / AVX10 | Ordinary vector implementation IDs, actual width and ISA-version requirements, per-operation tuning and OS-state checks. | Eligible only for the implemented FP64 instructions and verified OS support. |
| Intel AMX variants | Tile configuration/state lifecycle, operand-format descriptors, supported type tuples, and bounded block execution. | Resolve each concrete arithmetic variant; AMX presence alone is not FP64 support. |
| Joint AMD/Intel ACE | Type/primitive IDs, legal geometry, opaque layout IDs, and state requirements; add concrete movement/scaling metadata with a supported backend. | Published v1 matrix arithmetic is ineligible for FP64 accumulation. |
| A future FP64 matrix extension | Add concrete type/primitive/state descriptors and backend code, then tune it through the same matrix API. | Eligible only after the actual specification, build, runtime, and conformance checks establish it. |

**Execution-state lifecycle**

Keep capability inspection read-only. Model process-scoped permission, thread-scoped readiness, and per-call
state separately. Provide a preparation hook outside critical FFM only when a supported backend needs explicit
initialization; the current ABI must not grow a dummy permission service just to model a future accelerator.
The ordinary native wrapper owns ABI-required state setup/restoration; no tile state lives across Kotlin calls.

Linux AMX dynamic XSTATE permission is process-scoped, while first-use allocation belongs to a task and permission
interacts with signal-stack requirements. Future AMX/ACE code must implement its actual OS rules, not reuse an
SME thread flag. [Linux XSTATE usage](https://docs.kernel.org/arch/x86/xstate.html)

Describe `not_built`, `unsupported_cpu`, `unsupported_os`, `needs_enablement`, `ready`, and
`incompatible_context` with explicit reason codes. Denied preparation means ordinary fallback in AUTO and
failure in exact mode. Test the lifecycle through synthetic descriptors now; implement real enablement when
there is a supported arithmetic backend. Never change signal stacks or catch SIGILL to force eligibility.

**Bounded future extensibility**

Implement type tuples, opaque versioned layout IDs, legal geometry, state requirements, and unknown-record
handling now. These suffice to test a non-SME tile backend without introducing low-precision containers.
Keep packing and vendor-specific state inside the backend. Quantization, block-scale buffers, conversion
pipelines, and new public element families are added only with a concrete supported operation; reserve a
versioned extension seam rather than unused scale fields in every Double descriptor.

Keep ISA separate from CPU/vendor tuning. Reuse the same scalar-C and JVM-SIMD-C policy machinery for future
x86 kernels, with fresh measurements of setup, packing, and full-operation costs. Similar register widths or
tile dimensions do not justify importing Arm tuning values.

Add synthetic tests for ACE without FP64, AVX10 without ACE, unavailable/denied OS state, unknown feature
records, differing vendor profiles for one layout, and a future FP64 tile implementation. They exercise catalog
and planner decisions, never unsupported instructions. No speculative ACE/AMX implementation or new dtype API
is required by this project.
