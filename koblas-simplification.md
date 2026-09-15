# Koblas simplification plan

## Direction

Make Koblas a small Kotlin Multiplatform BLAS library backed by vendor implementations, with useful dense and
sparse Level 1 kernels and storage primitives. Vendor BLAS owns complete dense Level 2 and Level 3 operations.
The default implementation is selected once and cannot be replaced or configured at runtime.

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
| Optional vendor runtime module | Package oneMKL and AOCL BLAS binaries and their necessary runtime dependencies for JVM and supported Native targets |
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
Klause owns LU construction, hypersparse triangular solves, and Forrest–Tomlin updates already. HFactor is a
JVM test comparison dependency, not its production solver.

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

Use a closed vendor set and one readable selection function. Prefer Accelerate on macOS, AOCL on supported AMD
hosts, and oneMKL on supported Intel hosts. Check supported OS/architecture, ABI, and library availability once.
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

Linux ARM64 has no supported vendor in the proposed set. Decide whether to narrow accelerator support or add a
separately justified backend. This does not require dropping common storage/Level 1 support on that platform.
Do not pull AOCL-Sparse or vendor sparse matrix APIs into scope to retain unused sparse Level 2–3 operations.

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
Native integration. Runtime payload presence must not require provider registration. Fix the threading policy
explicitly, preserve the benchmark's one-thread comparison, and avoid changing global thread settings on each
operation. Do not claim single-thread behavior without checking the loaded vendor configuration.

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
Existing OpenBLAS comparisons may stay as a bench-only reference through the shared CBLAS transport; they do
not add OpenBLAS to production selection or require a second vendor binding implementation. Do not recreate the
same common primitive benchmark under every engine label.

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

## Implementation sequence: three PRs total

Use **one implementation PR in Koblas**, **one in Kumulant**, and **one in Klause**. These are three repositories,
so their changes cannot share a GitHub PR. Do not create separate foundation, packaging, benchmark, calibration,
or cleanup PRs. The steps below are work within these three PRs, not additional PRs. Split only if a concrete
integration/review blocker makes the combined change impractical; do not split merely by layer or source set.

| PR | Deliverable | Development dependency | Landing order |
|---|---|---|---|
| K — Koblas: `refactor: simplify vendor blas and benchmark routing` | Complete new API, all supported bindings, optional runtime bundle, Level 1/primitives, slim attributed bench, measured defaults, old code deleted | Start first; finalize contracts in step 1 | First, after all three candidate branches pass integration |
| U — Kumulant: `refactor: own cholesky on the simplified blas api` | Direct API migration, Cholesky and rank-one updates, solves/covariance, dependency update | Start after K's contracts are fixed; work alongside K and L | Second, using the accepted K artifact |
| L — Klause: `refactor: own sparse slice workflows` | Direct primitive/API migration, checked slices and pivot workflows, dependency updates | Start after K's contracts are fixed; work alongside K and U | Third, using accepted K and U artifacts |

The three PRs are one coordinated migration. Test candidate artifacts before landing, then land K → U → L with
matching dependency versions. K's merge does not upgrade existing consumer releases. Record the tested source
SHAs/artifact versions so dependency resolution cannot silently use stale or incompatible snapshots. Candidate
artifacts, a local repository, or supported composite builds provide integration; compatibility code does not.

### Step 1 — Fix contracts and capture usable evidence (K; sequential foundation)

1. Recheck current Koblas/Klause/Kumulant call sites, including Klause's transitive Kumulant use and in-repository
   HFactor callers. Record exact baseline SHAs and preserve the small scalar test oracles before deleting code.
2. Write the final operation/overload matrix: dense Level 1–3, sparse Level 1, generic primitives, and removals.
   Resolve GEMMT and other nonstandard APIs explicitly. Specify the common sparse contracts needed by Klause.
3. Fix the closed loading/fallback order, bundled-versus-installed precedence, target/ABI matrix, optional artifact
   coordinates and Native linking, threading, and missing-backend behavior. Record the Linux ARM64 decision and
   any changed numerical contracts. These decisions must be concrete before parallel binding work starts.
4. Define the small typed binding and call-route contract, including direct coverage, delegation, tails, no-work,
   unsupported cases, and buffer transfer reporting. Define owning/view signatures and ownership rules together.
5. Preserve useful fixture/report tooling and capture only comparable baseline workloads where attribution is
   established. Untrustworthy old labels cannot become baseline evidence merely because a timing exists.

**Exit:** final signatures and decisions are recorded here or in the PR's contract diff; independent work has
clear file ownership and shared contracts. No implementation depends on recreating the deleted SME plans.

### Step 2 — Build independent parts in parallel (K, U, L)

After step 1, the following tracks can proceed concurrently. Each track owns its implementation, tests, and
related documentation; coordinate changes to shared contracts through one integration owner.

| Track | Work | Waits for |
|---|---|---|
| K-A: JVM bindings | Direct FFM bindings, fixed JVM loading, symbol identity, memory transfers/workspace, complete public-call GEMM and submatrix proof | Step 1 ABI and window/route signatures |
| K-B: Native bindings and bundle | Native interop, fixed loading/linking, shared declarations, JVM/Native runtime payload packaging and dependency checks | Step 1 ABI, artifact layout, and loading contract; final JVM load test also needs K-A |
| K-C: Common API and Level 1 | Final containers/views/engine facade, scalar/SIMD dense and sparse Level 1, generic primitives, numerical/alias tests, direct call-site migration | Step 1 common contracts; full engine execution also needs K-A/K-B |
| K-D: Slim bench | Reuse capture/JMH/Native/report tooling, trim cases, binding-derived identities and exact-arm checks, deliberately misrouted tests | Step 1 route contract; real timings wait for K-A/K-B/K-C integration |
| U: Kumulant | Implement Cholesky/rank-one updates, migrate all old names/calls directly, update tests and docs | Step 1 dense signatures; execution checks wait for candidate K |
| L: Klause | Move checked slice/pivot workflows, call common primitives directly, update tests and docs | Step 1 primitive signatures; full integration waits for candidate K and U |

JVM/Native or individual vendor bindings may be developed independently under the same ABI. They still land in
one Koblas PR. Keep common signatures, shared declaration generation, Gradle coordinates, and the loader policy
under one owner; those files must not acquire competing designs. Parallel work does not require a registry,
temporary implementation of an old API, or an extra PR.

**Exit:** every final API has its implementation and call-route information; each track's focused checks pass.
Consumer code targets the new API directly. Work not executable until integration is recorded as unverified.

### Step 3 — Integrate, measure, and delete (K with U/L candidate integration; sequential gates)

1. Integrate K-A through K-D. Verify actual binding execution against route descriptions before trusting timing
   results. Check exact/default arms, whole-call scalar fallback, vector tails, views, and no-work paths.
2. Verify a complete GEMM/submatrix call on JVM and Native, including required transfers, ownership, and safe
   foreign-call behavior. Test installed/bundled/no-vendor configurations and the final threading policy.
3. Run the slim representative benchmarks and separate Level 1 crossover measurements. Hardware-specific runs
   can run in parallel on independent idle hosts. Serialize competing measurements on the same host. Apply only
   supported measured rules; record conservative behavior for unmeasured combinations.
4. Finish the immutable default, then delete the numerical C implementation, probe/catalog/profile machinery,
   portable Level 2–3, panel/tile/packed APIs, obsolete cases and C vendor wrappers. Delete obsolete overloads and
   forwarding paths too. Move independent scalar correctness oracles to tests; do not retain production copies.
5. Update every in-repository caller, including any affected HFactor adapter, API dump, test, and documentation
   directly. Keep HFactor algorithm changes out of scope. Remove stale installation flags, tuning keys, benchmark
   modes, and performance claims in this same PR. There is no later cleanup PR.
6. Build candidate K artifacts, then candidate U against K, then L against that exact K/U pair. Verify Cholesky
   rank-one updates remain O(n²), strict/regularized behavior and covariance extraction, and Klause's checked
   arithmetic and common-only integration. Run each repository's required checks on the final candidates.

**Exit:** the final three candidate branches integrate; the old architecture and all compatibility shims are
absent from K. Published-artifact tests, real route evidence, and representative measurement reports are attached.
Build success alone is insufficient when a required execution or attribution check is still missing.

### Step 4 — Review and land K → U → L (sequential)

1. Review the complete K change, including deletion completeness, symbol/call identity, numerical contracts,
   artifacts, and representative report rows. Resolve findings and rerun affected checks on the final head.
2. Run the required repository gates and verify required CI for each PR. Land K and make its tested artifact
   available; replace U's candidate dependency with that accepted version and verify U before landing it.
3. Update L to accepted K/U versions, verify the resolved dependency pair and required checks, and land L.
   Dependency bumps are included in U/L, not separate follow-up PRs. Never merge against an unavailable artifact.
4. Confirm all three final versions match the integration evidence. Report unsupported/unmeasured targets
   explicitly. Do not close the migration with deferred deletion, compatibility wrappers, or unresolved checks.

Implementation and focused tests can run in parallel as listed; contract decisions, shared integration,
measurement-before-default activation, and artifact/dependency landing are ordered. Do not edit `.github/`
without a separate request.

## Verification and completion

- Run `./gradlew :koblas:check :koblas-bench:check lintDocs` for implementation changes. Use full `./gradlew check`
  when changing HFactor or before pushing. Migrate the no-SIMD check to the new scalar/vendor behavior.
- Exercise installed and bundled libraries, missing libraries, missing Vector API, JVM and Native, supported
  vendor/OS combinations, independent exact instances, concurrency, and warmed allocation behavior. Include
  absent native-access permission, wrong ABI, incomplete libraries, and version/symbol identity checks.
- Verify numerical semantics, backing-buffer aliasing, offsets/strides, triangular and symmetric storage, and
  consumer integration. Follow each consumer repository's verification instructions when changing it.
- Reproduce exact-route attribution failures in tests before trusting crossover reports. Confirm representative
  report rows against actual bound calls, not just formatted names.
- Distinguish cross-build success, emulated correctness, real-hardware execution, and measured performance in
  support claims. Missing target evidence remains explicit; unrelated foundation work may proceed.
- Completion requires consumers migrated, old layers removed, benchmark attribution enforced, optional packaging
  verified, and documentation describing real support and missing-backend behavior. No compatibility shims,
  temporary bridges, legacy aliases/modes, duplicate public APIs, or production portable Level 2–3 remain.
