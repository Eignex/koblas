# Koblas simplification plan

## Direction

Make Koblas a small Kotlin Multiplatform BLAS library backed by vendor implementations, with useful dense and
sparse Level 1 kernels and storage primitives. Vendor BLAS owns complete dense Level 2 and Level 3 operations.
The default implementation is selected once and cannot be replaced or configured at runtime.

**Every BLAS invocation uses one compute thread; the bindings remain concurrently callable.** A function must not
split work on its input across multiple threads. Applications may call the same immutable binding from multiple
threads at once, including with shared read-only inputs. This applies to oneMKL, AOCL, ArmPL, Accelerate, and every
benchmark arm. The restriction is per invocation, not a process-wide limit of one active BLAS call.
Different invocations may run in parallel. This behavior is static: no public API, constructor option, per-call
flag, system property, environment override, or runtime configuration may enable internal multithreading or
change the BLAS thread count.

This is the sole architecture and implementation plan. The former SME plans are deleted; their useful numerical,
storage, attribution, and measurement requirements are incorporated here. No SME/SME2 or future-ISA work remains.

**Backward compatibility is explicitly unimportant. No compatibility shims, including temporary ones.** Source,
binary, configuration, old C ABI, engine, and packed-format compatibility may all break. Do not add deprecated
aliases, forwarding entry points, legacy modes, dual public APIs, or bridges to keep old callers compiling.
Change callers directly. Delete replaced APIs and implementations in the same PR as their replacements, with
matching API dumps, tests, and documentation. Intermediate work on a branch may fail to compile while the direct
migration is in progress; the integrated PR must pass. Preserve required numerical and storage behavior, and
explicitly decide any contract changes before deleting its implementation.

## Scope and ownership

| Component | Responsibility |
|---|---|
| `koblas` | Public containers/views, validation, workspace, dense BLAS bindings, dense/sparse Level 1, generic sparse primitives, immutable selection |
| Optional vendor runtime module | Package oneMKL, AOCL, and ArmPL BLAS binaries and their necessary runtime dependencies for JVM and supported Native targets |
| `klause` | Sparse slice workflows, checked solver arithmetic, pivot policy, LU construction, hypersparse solves, Forrest–Tomlin updates |
| `kumulant` | Cholesky factorization, rank-one factor updates, regularization policy, solves and covariance extraction built on Koblas |
| `koblas-bench` | Slim operation inventory using production bindings, preserving the useful existing measurement and reporting machinery |

Bindings belong to the base library and work with compatible installed libraries as well as the optional
bundle. Accelerate is supplied by macOS and gets no binary bundle. Reuse ABI declarations and adapter semantics
between JVM and Native; bind vendor entry points directly through thin FFM and Native interop implementations.
Share the declaration source and common validation/marshaling contracts, not a new C forwarding shim.

Remove:

- Bundled numerical C kernels and their build/probe/catalog machinery.
- Production portable dense and sparse Level 2–3 implementations.
- Public panel kernels, packed panels, tile kernels, and the associated composition and tuning machinery.
- Extensible host/provider registration, configurable performance profiles, startup tuning, and custom ISA work.
- Benchmark cases whose only purpose is comparing removed implementation layers.

Keep small scalar correctness oracles in tests, and production scalar Level 1/generic storage primitives where
needed. Retaining a test oracle does not retain a production Level 2–3 fallback.

The initial dense matrix binding surface is the existing standard operations: Level 2 GEMV, SYMV, GER, SYR,
SYR2, TRSV, TRMV; Level 3 GEMM, SYMM, SYRK, SYR2K, TRMM, TRSM. Audit nonstandard operations such as GEMMT
separately for direct availability or explicitly named composition. Storage transforms and Level 1 extensions
such as sum/SSQD may remain Kotlin implementations; do not invent a vendor entry point or claim vendor coverage
for them. Audit mixed dense/sparse overloads too: preserving a generic signature must not leave a hidden sparse
Level 2–3 algorithm behind.

## Consumer requirements

### Klause

The source audit of `/home/rasmus/Workspaces/klause` at `87181fd43` found no calls to Koblas sparse Level 2–3.
Klause owns LU construction, hypersparse triangular solves, and Forrest–Tomlin updates already, and depends on
no Koblas factorization or basis artifact.

Preserve common APIs for CSC construction and column iteration, triplet compression, indexed dot/norm/scatter/
AXPY, dense AXPY, sparse accumulation, touched-index gather/compact/clear, masked absolute maximum, and scratch.
Move `SparseSlices` solver workflows into Klause, retaining their numerical behavior. In particular, checked
scatter and ordered dot reduction detect nonfinite arithmetic and product underflow; ordinary vendor BLAS is
not a substitute for that contract. Keep pivot eligibility and tolerance policy in Klause.

Generic primitives must be usable from `commonMain` without platform code in Klause. Define ordering, repeated
indices, exact-zero compaction, aliasing, mask behavior, and nonfinite values explicitly. Preserve allocation-free
reuse of caller-owned buffers. Develop the consumer migration against the replacement API in parallel, and verify
the integrated candidate versions before landing. Old releases remain on their existing dependencies; the new
Koblas artifact must not retain their entry points.

### Kumulant

The source audit of `/home/rasmus/Workspaces/kumulant` at `31a3802` found an older Koblas API and no direct public
panel/tile calls. Bayesian regression needs strict and regularized Cholesky, rank-one factor updates, triangular
solves/products, SYRK, and covariance extraction. Its observation update must remain O(n²); do not replace its
rank-one update with O(n³) refactorization per observation.

Implement ordinary blocked Cholesky in Kumulant: a small Kotlin diagonal factorization, vendor SYRK/GEMM updates,
and TRSM block solves. Preserve matrix offsets and leading dimensions so blocks can be passed without copying
whole submatrices. Public tile kernels are unnecessary. Small-input overhead and rank-one update performance
must be measured; accepting their removal is not a measured claim that performance is unchanged.
This follows the ordinary blocked structure used by
[LAPACK DPOTRF](https://netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f_source.html).

## Static vendor selection

Use a closed vendor set and one readable selection function. Prefer Accelerate on macOS, Arm Performance Libraries
(ArmPL) on Linux ARM64, AOCL on supported AMD x64 hosts, and oneMKL on supported Intel x64 hosts, and end every
Linux order with OpenBLAS so a host carrying only the distribution's BLAS computes instead of raising. Check OS and
architecture before CPU vendor; an ARM processor from an unfamiliar vendor still takes the ArmPL route.
Check supported OS/architecture, ABI, and library availability once.
Define a short, fixed compatible fallback order before implementation; do not infer compatibility from successful
library loading alone. The guarantee is a preferred available implementation, not the fastest possible function
on every input.

- Fix one ABI initially: double precision, column-major matrices, 32-bit BLAS integers matching Kotlin `Int`.
- Choose fixed supported library names/locations and deterministic bundled-versus-installed precedence.
- Resolve symbols against the selected library, not a process-global search that can bind the same CBLAS name
  from another loaded vendor. Isolate benchmark vendors in separate processes where their runtimes conflict.
- Keep library handles alive for the lifetime of their calls. No per-call loading or symbol lookup.
- Keep containers, Level 1, and generic primitives usable without a vendor library. Accelerator-dependent calls
  fail clearly when no supported vendor exists; default initialization must not prevent unrelated operations.
- Expose explicit immutable implementations to tests/bench without replacing process-global state.
- Use only measured, checked-in Level 1 crossover rules where JVM Vector API or scalar execution wins.
- Do not reuse thresholds measured for the removed C kernels.

### Single-threaded calls and concurrent bindings

- Use sequential vendor libraries where available. Otherwise establish the vendor's supported single-thread
  configuration once before its first arithmetic call, including disabling dynamic thread expansion as needed.
  A backend/configuration that cannot enforce single-threaded execution is unsupported.
- Expose no thread-count setting, parallel execution policy, worker pool, or multithreaded algorithm. Do not
  change process-global thread settings around individual operations or enable vendor defaults that add workers.
  The fixed single-thread configuration is an implementation detail established during initialization, not a
  configurable default. Do not expose even a startup-only override. Vendor configuration must not silently
  override this invariant.
- Package only dependencies required for the selected single-thread implementation. Do not add a parallel
  runtime merely to offer multithreaded execution.
- Bench runs each operation with one compute thread. No multithreaded benchmark modes, throughput scaling
  suites, or calibration against multithreaded vendor calls. Verify and record the effective configuration;
  fail a selected benchmark arm if it cannot establish single-threaded execution.
- Bindings must be reentrant and callable from any application thread. Concurrent calls through the same binding
  are supported; do not serialize all arithmetic behind a global lock or require a designated caller thread.
  Shared read-only inputs are allowed. Each in-flight call owns its mutable scratch/workspace; callers must
  synchronize overlapping writes or writes overlapping another call's reads. Per-call alias rules still apply.
- Test concurrent binding calls, shared read-only inputs, distinct outputs/workspaces, and safe initialization
  and lifetime. These are binding safety tests, not multithreaded execution of a single BLAS function. Parallel
  implementation tracks and measurements on independent hosts also remain allowed.

ArmPL supplies the Linux ARM64 dense BLAS backend on both JVM and Native, using the same common CBLAS declarations
and 32-bit BLAS integer contract. Include its required runtime payload in the optional bundle and its exact arm
in bench. Accelerate remains the macOS ARM64 default. This adds no new OS target or ISA-specific kernel code.
Use the chosen ArmPL release's documented ABI, library names, runtime dependencies, and redistribution terms;
do not assume another vendor's packaging or threading details apply.
[ArmPL installation and supported Linux platforms](https://learn.arm.com/install-guides/armpl/).

Do not bind ArmPL's LAPACK, FFT, or sparse matrix interfaces, or pull AOCL-Sparse into scope to retain unused
sparse Level 2–3 operations. A missing ArmPL installation and absent optional bundle use the same explicit
missing-backend behavior as other platforms; Linux ARM64 accelerator support is part of this migration.

## Storage, numerical behavior, and foreign calls

Use one common matrix-window boundary: one validation/adaptation path for owning matrices
and views, retaining offsets, logical dimensions, vector strides, and physical leading dimensions. No interface
default body may quietly send a view through a different scalar algorithm. For BLAS-compatible windows, preserve
direct submatrix addressing. Make any necessary JVM heap/native transfer or nonrepresentable-stride staging
explicit; never promise zero-copy FFM merely because the logical view is zero-copy.

Before replacing each family, inventory its existing tests and classify the contract:

- Preserve shape/buffer checks, permitted aliases, untouched backing storage, selected triangles, implicit unit
  diagonals, robust norms, and alpha/beta/empty-operation no-read behavior. Staging must obey these rules too:
  beta zero does not justify copying old output, and an implicit diagonal must not be read during conversion.
- Compare finite arithmetic against an independent scalar oracle with appropriate tolerances. Test NaN/Inf,
  extreme magnitudes, subnormals, and signed zero according to the declared operation contract.
- Identify stronger existing ordered/overflow behavior that vendor BLAS cannot guarantee. Resolve each mismatch
  explicitly through validation, the new operation's semantics, a documented contract change, or removal; do not retain
  the portable Level 2–3 implementation covertly to satisfy an old test. Solver-specific checked arithmetic stays
  in Klause and keeps its stronger semantics.
- Reject invalid/unsupported routes before output mutation. Never retry an operation through another backend
  after partially updating the destination. Check dimension, offset, byte-size, and workspace arithmetic for
  overflow before allocation or foreign calls.

Whole vendor BLAS calls change the JVM memory boundary. Do not carry the old critical-downcall strategy across
unchanged: arbitrary vendor operations may take a long time, allocate, or coordinate worker threads. Start with
ordinary non-critical FFM and native buffers where required, with explicit copy-in/copy-out and reusable scratch
lifetime. Any critical-call exception needs a documented bounded leaf satisfying the
[JDK critical-call requirements](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/foreign/Linker.Option.html#critical(boolean));
small dimensions alone do not prove that an opaque vendor call qualifies. Do not reintroduce a Kotlin BLAS block
scheduler just to make whole vendor operations fit the old call boundary.

No foreign pointer outlives its valid array pin or buffer owner. Pin Native operands for the call, keep mutable
scratch exclusive to each concurrent operation, and test cleanup on exceptions. Retained native storage is a
separate API decision only if measurements justify it; silently caching copies of mutable arrays is invalid.
Measure transfer-inclusive throughput, allocation after warmup, and GC/safepoint responsiveness. Preserve the
inline requirement for JVM helpers returning `DoubleVector` and the existing uninstrumented allocation checks.

Package only the required vendor runtime components. Verify target architecture, ABI, dependent libraries,
redistribution notices, and extraction/link behavior for the published JVM and Native artifacts. A Native
optional module must provide a tested link/package integration; JVM classpath discovery does not demonstrate
Native integration. Runtime payload presence must not require provider registration. Enforce the single-threaded
execution requirement in production and bench, and verify it against the actual loaded vendor configuration.

## Benchmark implementation identity

Each implementation must report what it implements directly and where a particular call will execute.
An engine name, platform label, or requested benchmark arm is not evidence of the function's implementation.

The production routing decision and its description must have one source of truth. Bench must consume that
description for the exact operation, overload, dimensions, strides, flags, and other route-relevant arguments.
Keep this as a small closed contract attached to bound functions, not a registry, planning language, or generic
execution graph.

Expose two related facts through the common binding surface: the operations directly implemented by a layer,
and the route of a concrete call. An implemented operation can still decline particular strides or dimensions;
the operation list alone cannot authorize a benchmark case. Attach identity to the same typed binding/callable
used for execution. Default methods and delegated helpers must propagate the callee's identity rather than
inherit the receiver's name. Require every retained entry point and overload to provide coverage information.

- Distinguish direct implementation, delegation, composition, semantic no-work, and unsupported calls.
- Report JVM versus Native, requested layer, actual implementation/vendor, resolved entry point, and any adapter
  or delegation involved. Vendor CPU-internal dispatch remains opaque unless the vendor exposes reliable data.
- Exact comparison arms execute only their own implementations. If SIMD dot delegates to scalar for a case, that
  case is not a SIMD measurement. If a vendor lacks a sparse primitive, do not time a Kotlin substitute as vendor.
- A SIMD loop's normal scalar remainder is part of its implementation, and should be described as such. A call
  taking only the scalar path is not evidence of SIMD execution. Distinguish whole-call delegation from a tail.
- The default policy may delegate; report the actual destination. Composed helpers are explicit end-to-end cases,
  never mislabeled as direct implementation comparisons.
- Resolve and validate identity before timing. Descriptions must not allocate or trace inside every timed call.
- Default-policy timings must still call the normal production entry point and include its dispatch/adaptation
  cost. Inspecting its route must not replace it with a prebound raw leaf and silently remove that overhead.
- Unknown or ambiguous destinations are not valid exact comparisons. Unsupported cases have a reason and no
  timing; a missing explicitly requested backend or an unexpected execution mismatch fails the capture.
- Correctness tests must verify the called implementation as well as the description, including threshold,
  stride, overload, aliasing, and no-work cases. A hardcoded label alone is insufficient.

### Minimal reporting and enforcement

Extend the existing `CaseWork`/measurement plumbing so the timed work carries its binding-derived route. Remove
post-hoc attribution from mode/case strings: `Runner.measurement` currently calls `actualPackedKernel` separately
from `CaseWork.run`, and JVM/Native packed selection also differs. Their replacement must not recreate two
independent answers to which implementation is timed.

Keep a compact record: requested implementation, resolved implementation and entry point, direct/delegated/
composed/no-work/unsupported classification, adapter or delegate names when present, and eligibility reason.
Associate vendor identity with the loaded artifact recorded in metadata. Format it in the existing report or
add only fields required for filtering/aggregation. Neither function addresses nor verbose host descriptors
belong in case IDs. No generic tracing framework or catalog schema is needed.

| Example call | What bench does |
|---|---|
| Exact JVM SIMD dot using a vector loop plus scalar tail | Time as the direct JVM SIMD implementation; describe the tail |
| Exact JVM SIMD dot whose tiny-input or stride path delegates wholly to scalar | Skip the exact SIMD comparison with a reason |
| Default dot choosing scalar or SIMD by size | Time the normal default call and record that actual choice |
| JVM oneMKL GEMM with native-buffer transfers | Record oneMKL DGEMM plus JVM staging; include staging in the public-call timing |
| JVM or Native Linux ARM64 GEMM | Record the resolved ArmPL CBLAS entry point and any runtime-specific transfers; do not infer its internal ISA |
| Vendor arm for a Kotlin-only accumulator or compaction primitive | Unsupported for that arm; benchmark its actual Kotlin implementation once |
| Alpha-zero operation that only scales the destination | Report the scaling implementation; do not claim the vendor product ran |
| True empty/no-op call | A semantic overhead case only, not accelerated arithmetic evidence |

Add a deliberately misrouted test fixture to prove that exact-arm enforcement catches delegation. Verify real
dispatch with test-only recording/counting implementations or untimed diagnostic calls at the bound-call
boundary, and separately verify vendor symbols resolve to the intended library. Exercise owning/view overloads,
threshold boundaries, entire scalar fallbacks, vector tails, and missing symbols. Do not add counters to measured
hot loops. Unknown/data-dependent routes stay out of exact comparisons until their actual branches can be
established; preflight fixtures must match the values and resets used during measurement.

Retain `cases.txt`, fixture recipes, suite/operation/smoke filters, JVM JMH, Native calibration, warmups, forks,
hardware-keyed reports, CSV aggregation, and preservation of the previous report when a selected target fails.
Use the same production bindings on both runtimes instead of a separate vendor wrapper in the C runner.
Keep sparse Level 1/generic primitive cases and representative dense Level 1–3 cases; remove tile/packing and
unused sparse matrix cases. Keep narrow opt-in sweeps for actual Level 1 crossover decisions.
Existing OpenBLAS comparisons stay as the reference arm through the shared CBLAS transport, and do not require
a second vendor binding implementation. OpenBLAS is also the last resort in production selection on Linux,
after every tuned vendor: a host that has only the distribution's BLAS computes instead of raising, and the
route names the library that ran. Do not recreate the same common primitive benchmark under every engine label.

Preserve case identity and timing boundaries for comparable workloads. Full public-call measurements include
required FFI and storage adaptation. Arithmetic-only measurements must say which setup/reset/copy costs they
exclude. Record resolved library versions and actual threading configuration in report metadata. A route change
between forks must not be silently averaged into one result.

Preserve logical input/flag/stride/alpha/beta definitions independently of backend defaults. A changed fixture
or timing boundary needs a new baseline unless equivalence is demonstrated; matching operation names alone is
insufficient. Keep source SHA and report provenance rather than adding a schema-migration system. Exact, default,
and composed results are separate comparisons; failed attribution is never reported as a performance win.

Calibrate scalar-versus-vendor and JVM-SIMD-versus-vendor separately, including transfer costs. Record case/source
SHA, hardware, vendor/version, JDK/compiler, report, and measured range for each checked-in crossover. Validate
on repeated runs and held-out lengths/strides; prefer a simpler rule when results overlap. Unmeasured hardware
gets a documented conservative choice, not a claim of calibration. A raw leaf win cannot justify a slower public
call. Do not add runtime overrides or an importer/configuration language to preserve the old tuning system.

## Implementation sequence: four Koblas PRs and two consumer PRs

Use **four implementation PRs in Koblas**, **one in Kumulant**, and **one in Klause**: six total. Split the Koblas
change at working subsystem boundaries, keeping JVM/Native support, tests, API dumps, docs, and deletion of the
replaced subsystem together. No compatibility shims or separate cleanup PR. An untouched old subsystem can
continue operating until its named cutover PR; do not bridge it to the new API or expose duplicate public APIs.

| PR | Deliverable | Prerequisites for landing |
|---|---|---|
| K1 — `feat: add shared vendor blas bindings` | Final binding/route contracts, installed-library loading for oneMKL/AOCL/ArmPL/Accelerate on JVM and Native, safe memory calls, exact binding checks and bench attribution foundation | Contract decisions below |
| K2 — `refactor: reduce sparse support to level one primitives` | Final common sparse primitives and Level 1 routing, slice workflow removal, sparse Level 2–3 deletion, matching slim sparse bench | K1; Klause candidate verifies the changed primitive/slice contract |
| K3 — `refactor: route dense blas through vendor bindings` | Whole-operation dense cutover, final engine and views, scalar/SIMD Level 1, full slim bench migration, removal of numerical C/panel/tile/packed/dispatch machinery | K1 + K2; Kumulant candidate verifies the new dense API |
| K4 — `feat: bundle vendor runtimes and calibrate defaults` | Optional oneMKL/AOCL/ArmPL runtime payloads, artifact tests, measured Level 1 choices, final consumer and performance evidence | K3; trusted attribution before calibration |
| U — `refactor: own cholesky on the simplified blas api` | Direct Kumulant migration, Cholesky/rank-one updates, solves/covariance, dependency update | Final K4 artifact available |
| L — `refactor: own sparse slice workflows` | Direct Klause migration, checked slices/pivot workflows, dependency updates | Final K4 and U artifacts available |

### Before parallel work — fix the contracts in K1

1. Recheck Koblas/Klause/Kumulant call sites. Record baseline SHAs, preserve independent scalar test
   oracles, and retain only baseline timings whose inputs, timing boundary, and executed implementation are known.
2. Record the final operation/overload matrix and common primitive signatures. Decide nonstandard APIs such as
   GEMMT and any stronger numerical contracts that vendor BLAS cannot provide.
3. Fix the four-vendor loading/fallback order, supported targets, LP64 ABI, artifact coordinates/layout,
   bundled-versus-installed precedence, single-thread enforcement, and missing-backend behavior. ArmPL owns Linux ARM64;
   Accelerate owns macOS ARM64. K4 supplies optional payloads to this final loader without provider registration.
4. Define the small typed binding/route contract and final owning/view signatures. Assign one owner to shared
   declarations, common signatures, Gradle coordinates, and loader policy before others build against them.

**Exit:** executable binding work and consumers can use fixed contracts. This is part of K1, not a planning PR.

### K1 — Shared vendor bindings and exact identity

1. Implement direct CBLAS bindings for oneMKL, AOCL, ArmPL, and Accelerate on the supported JVM/Native targets.
   Share declarations and common argument rules; keep symbol resolution and pointer handling platform-specific.
2. Implement fixed installed-library loading and the final bundle lookup convention. Verify ABI, library/symbol
   identity, missing/partial libraries, independent explicit instances, safe memory lifetimes, and single-thread
   enforcement for every supported vendor. Unsupported threading configurations must not enter arithmetic.
3. Prove whole GEMM and submatrix calls with transfer-inclusive JVM execution and Native pinning. Cover Level 1,
   structured and triangular signatures too; do not defer ABI errors until the dense default switches.
4. Attach direct coverage and concrete route information to actual bound calls. Add deliberately misrouted tests
   and the minimal existing-bench plumbing needed to exercise exact vendor calls. Migrate retained vendor cases
   from the separate C wrapper directly; delete each replaced wrapper in this PR. K2/K3 trim their removed cases.

**Exit:** installed vendors are callable and attributable on both runtimes; exact requests cannot substitute
another layer. Existing dense orchestration remains untouched until K3. New bindings are final implementation
components, not a compatibility surface. Real execution evidence is separate from cross-build evidence.

### K2 — Common sparse primitives and sparse removal

1. Implement the final common accumulator, compression/compaction, touched-index, and masked-reduction APIs.
   Keep caller-owned storage, duplicate/order/zero/nonfinite semantics, and allocation-free reuse explicit.
2. Route sparse Level 1 through its final scalar/SIMD or available vendor implementation with truthful per-call
   coverage. Remove replaced indexed C paths directly; exact vendors do not inherit Kotlin-only operations.
3. Move solver-specific `SparseSlices` workflows to the L candidate branch and delete their Koblas entry points.
   Delete sparse Level 2–3, prepared sparse machinery, and mixed overloads that preserve those algorithms.
4. Migrate in-repository callers and sparse bench cases directly. Keep one benchmark for a common primitive,
   validate actual routes, and remove sparse matrix and solver-workflow benchmarks from Koblas.
5. Verify L's checked arithmetic/primitive integration against the candidate K2 artifact before landing K2.
   Full L/Kumulant integration waits for K3/K4; this focused check does not claim that integration is complete.

**Exit:** Koblas's sparse surface is storage, Level 1, and generic primitives. No sparse Level 2–3, old slice
entry points, or forwarding compatibility code remain. Dense internals still needed by K3 are untouched.

### K3 — Dense cutover and removal of the kernel architecture

1. Replace dense Level 2–3 with whole vendor operations and use one validation/memory path for owning matrices
   and views. Preserve offsets/leading dimensions, permitted aliases, no-read rules, and explicit transfer costs.
2. Finish the immutable engine using the fixed vendor selection and final scalar/JVM SIMD Level 1 components.
   Start with explicit conservative Level 1 choices; K4 adds only crossovers supported by the new measurements.
3. Complete the slim benchmark migration to production calls and binding-derived routes. Preserve capture,
   JMH/Native calibration, fixtures, filters, and report machinery. Verify exact/default distinctions and all
   retained overloads before accepting timings. Remove tile, panel, packing, and obsolete engine cases.
4. Delete production portable dense Level 2–3, numerical C kernels/build tasks, probes/catalogs/profiles, packed
   and panel/tile APIs, policy wrappers, old tuning keys, and remaining obsolete C vendor-runner code. Move small
   scalar correctness oracles to tests.
5. Update API dumps, tests, docs, installation flags, and performance claims in this same PR. Verify U's Cholesky,
   rank-one updates, solves, regularization, and covariance against the candidate K3 artifact before landing.

**Exit:** the replacement architecture and slim attributed bench work with installed vendors. All superseded
kernel machinery and compatibility code are gone; K4 has no deferred cleanup responsibility.

### K4 — Runtime bundle, calibration, and final integration

1. Package the required oneMKL, AOCL, and ArmPL runtime payloads in the optional module. Test actual JVM artifact
   extraction/loading and Native link/package behavior, target/ABI/runtime dependencies, single-thread execution,
   and required notices. Select sequential payloads where available; do not bundle optional multithread modes.
   Accelerate continues to use the system framework. Exercise installed, bundled, and missing-vendor paths.
2. Reconfirm binding-derived attribution and effective single-thread execution on each measurement host. Run the slim dense
   and sparse Level 1 suites plus representative dense Level 2–3, including Linux ARM64 ArmPL on JVM and Native.
3. Measure scalar-versus-vendor and SIMD-versus-vendor independently, including transfers and held-out cases.
   Check in only justified fixed Level 1 rules with provenance. Preserve conservative choices for unmeasured
   combinations; do not infer ArmPL performance from another backend or from emulation.
4. Build candidate K4 artifacts, then U against K4, then L against that exact K4/U pair. Verify numerical behavior,
   O(n²) rank-one updates, checked sparse arithmetic, common-only Klause integration, and resolved dependencies.
   Run each repository's required checks on the final candidates and review the complete artifact/report evidence.

**Exit:** optional artifacts, measured choices, and both migrated consumers are verified. Required missing
hardware or integration evidence is explicit unfinished work, not a reason to claim completion.

### Parallel work and ordered gates

| Work that can run in parallel | Earliest start | Must wait for |
|---|---|---|
| K1 JVM and Native/vendor implementations | K1 contracts fixed | One shared declaration/loader design; combined ABI and call tests before K1 lands |
| K2 sparse primitives/removal and K1 binding implementation | Common signatures and route contract fixed | K1 lands before K2; L's focused sparse migration check |
| K3 dense wrappers, engine migration, and bench work | K1 contracts fixed | Executable K1 bindings and accepted K2 for integration/landing |
| K4 runtime packaging for each vendor/platform | K1 artifact layout and loader fixed | Final K3 runtime for artifact verification; no early measured defaults |
| U Cholesky and L slice migration | Final dense/primitive signatures fixed in K1 | Candidate K2 for L's primitive tests; candidate K3 for U's dense tests; K4/U/L for final integration |
| Vendor/hardware benchmark runs | K3's attribution checks pass and candidate K4 is stable | Independent idle hosts; serialize measurements that compete on one host |

Development can overlap; land **K1 → K2 → K3 → K4 → U → L**. Rebase stacked work onto accepted prerequisites and
rerun affected checks after integration. Shared contract decisions, actual-route verification before calibration,
calibration before measured-default activation, and artifact publication before consumer dependency landing are
ordered. Do not have independent tracks invent separate route descriptions or mutate shared Gradle/API files
without coordination.

Use candidate artifacts, a local repository, or supported composite builds for pre-merge integration. Record
source SHAs and artifact versions; never rely on a stale mutable snapshot. Existing consumer releases stay on
their old dependencies while candidate branches migrate directly, so no shim is needed. Review each PR's final
head and verify its required checks before landing. Publish the accepted K4 artifact, land U against it, then
land L against accepted K4/U versions; include dependency bumps in U/L. Do not edit `.github/` without a separate
request. No separate cleanup, compatibility, or dependency-bump PR is planned.

## Verification and completion

- Run `./gradlew check lintDocs` for implementation changes. Migrate the no-SIMD check to the new scalar/vendor
  behavior.
- Exercise installed and bundled libraries, missing libraries, missing Vector API, JVM and Native, supported
  vendor/OS combinations, independent exact instances, concurrency, and warmed allocation behavior. Include
  absent native-access permission, wrong ABI, incomplete libraries, and version/symbol identity checks.
- Verify numerical semantics, backing-buffer aliasing, offsets/strides, triangular and symmetric storage, and
  consumer integration. Follow each consumer repository's verification instructions when changing it.
- Reproduce exact-route attribution failures in tests before trusting crossover reports. Confirm representative
  report rows against actual bound calls, not just formatted names.
- Verify one compute thread per BLAS invocation for every supported vendor and packaged artifact, plus concurrent
  calls through shared bindings. Internal multithreaded arithmetic, configurable BLAS thread counts, and
  benchmarks that split one invocation across workers fail the acceptance criteria.
- Distinguish cross-build success, emulated correctness, real-hardware execution, and measured performance in
  support claims. Missing target evidence remains explicit; unrelated foundation work may proceed.
- Completion requires consumers migrated, old layers removed, benchmark attribution enforced, optional packaging
  verified, and documentation describing real support and missing-backend behavior. No compatibility shims,
  temporary bridges, legacy aliases/modes, duplicate public APIs, or production portable Level 2–3 remain.
