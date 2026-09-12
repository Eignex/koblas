# Koblas SME implementation: PR-by-PR transition

Status: implementation is tracked in the individual sessions and GitHub PRs; planning IDs below are not GitHub numbers.

Architecture: [koblas-sme.md](koblas-sme.md). Keep one backend-neutral portable Level 2/3 orchestration per
operation, with reusable strategies and backend-specific block kernels. Do not duplicate portable BLAS by ISA.

The target is independently selectable SME and SME2 kernels, an extensible C probe, explicit ordinary SIMD
widths, operation-level planning, layout-bearing packed operands, measured runtime-specific defaults, and seams
for future AMD/Intel AVX10, AMX, and ACE. Existing APIs and implementation structure may change.

**Backward compatibility is not required in any PR.** Choose the optimal final architecture even when it breaks
existing public/internal Kotlin APIs, C ABI and symbols, packed formats, engine selection, or configuration keys.
Existing consumers may need to rebuild and migrate. Do not preserve source, binary, configuration, or packed-data
compatibility through deprecated aliases, legacy modes, dual APIs, or permanent shims. Mathematical correctness,
required alias/no-read semantics, and supported operation coverage remain requirements.

**Scope discipline**

Build both SME backends and the shared dense interfaces. Existing sparse callers migrate with the panel API;
new sparse-dense analysis is a measured follow-up. Standalone vector experiments belong to calibration, with
no requirement to ship new streaming leaves. Native-memory workspaces and new vendor bindings are added only
when measurements justify them. Future ACE/AMX support requires extensible descriptors and synthetic tests,
not speculative low-precision metadata or a dummy OS-enablement implementation.
Support logical modes through shared packing/scheduling where possible; do not multiply native implementations
across every ISA/layout/transpose/tile combination without measured need.

**Consolidation and current progress**

This sequence has **14 PRs**, covering all 27 original steps. `W01`–`W27` are stable work-package IDs matching
those original planning numbers; they are not GitHub PR numbers. The mapping below assigns every package once.
Each package retains its transition, implementation scope, technical verification, and exit criteria. A merged
PR gets one goal, one implementation session, one independent review, and one final green-CI completion gate;
its reviewer covers the union of the package-specific risks. Checks need not be repeated without relevant changes.

PR 01 has already landed as [GitHub PR #540](https://github.com/Eignex/koblas/pull/540), merged on 2026-09-12 at
`1be759ff`. Keep W01 as the acceptance record; audit existing results and carry forward missing target evidence
instead of reopening or reimplementing it. Its follow-up simplifications use explicit named packed recipes,
source-SHA provenance, and existing case/run/sample records rather than separate version labels or duplicated
physical settings. New work begins with PR 02 after checking the accepted baseline.

Within merged PRs, update producers and consumers directly; do not introduce an adapter merely to bridge two
work packages landing together. Temporary adapters are only for consumers in a later PR. Build the simple
concrete implementation first: static native descriptors, typed tables, ordinary selection functions, and
validated operands. Preserve every operation/mode and correctness requirement without creating a distinct
interface, packing format, or specialized kernel for every combination. oneMKL parity remains a separate goal.

**How to land the transition**

- Land in numbered order. The prerequisite column names the essential technical dependencies; every PR also
  rebases onto the accepted preceding state. These labels are planning IDs, not GitHub issue/PR numbers.
- Each merge must compile, pass its applicable checks, and leave all existing operations callable. If a backend
  does not yet implement an operation, select an explicitly identified available implementation. Exact kernel
  execution must never silently substitute another backend.
- New ISA kernels are initially exact-selectable. AUTO eligibility requires correctness and target measurements;
  PR 13 installs the complete measured policy. Shared orchestration can migrate earlier using conservative
  available-kernel profiles.
- Temporary adapters exist only to cross a transition boundary. They do not define the final ABI or packing
  format, and each has a removal owner in this document. Do not publish them as new public APIs. Keeping each
  intermediate PR buildable is an integration requirement, not a backward-compatibility commitment. Remove an
  adapter as soon as its last in-repository consumer migrates; prefer updating callers directly in the owning PR.
- Update relevant API dumps, KDoc, and test callers in the PR that changes an API. PR 14 finishes migration
  documentation; it is not permission to leave intermediate API checks broken.
- Do not split PRs solely by source set when that leaves JVM or Native broken. A kernel contract change includes
  the bindings and reference implementation needed to exercise it.
- Suggested titles below use single-line Conventional Commits without scopes. Actual PR descriptions follow
  the repository template and refer to real issues only; do not fabricate `Closes` numbers.
- Launching a step authorizes its session to implement, commit, push, OPEN a GitHub PR, and fix it until green.
  A local patch or commit alone is not completion. Do not merge automatically or launch the next step.

**Fresh sessions and model settings**

Every numbered PR starts in a fresh implementation session and ends with an independent review in a separate
fresh session. Do not continue the next PR in the previous PR's conversation, fork the implementation history
into the reviewer, or count implementation-session self-review as independent review. Using the same model in
two separate sessions is allowed; independence comes from separate context and independently checking evidence.
Implement each merged PR as its ordered work packages within one implementation session and one reviewed PR.
A work package is not a separate PR/session gate. Split only if a concrete review/build risk warrants it; preserve
all package requirements and the same session/goal/review/CI rules for each resulting PR.

Use GPT-6 Astra (`gpt-6-astra`) for implementation and review. The official model documentation identifies it
for complex coding/reasoning and lists `high` and `xhigh` as supported reasoning settings.
[Model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra).
The assignments below are engineering recommendations for this project, not measured model comparisons:
`high` for the existing baseline step (PR 01); `xhigh` for the remaining consolidated
implementation work and every independent review. Settings are stated explicitly in every PR section.
At session start, confirm availability and record the actual model/effort; if unavailable, revise the assignment
explicitly rather than silently substituting another model. Model settings never replace verification gates.

1. Start one fresh implementation session per PR on a fresh `codex/` branch from the accepted preceding state.
   Read AGENTS.md, both plans, prerequisite handoffs, and the affected code. Create an actual session goal with
   `create_goal`, using the Goal stated in that PR's section. Do not set a token budget unless the user requests
   one. If the session already has the same active goal, continue it instead of creating a duplicate.
2. Implement this PR and run its applicable local gates, including the full repository check before pushing.
   Commit, push the branch, and OPEN a GitHub PR against the correct base once the change is reviewable. A draft
   is acceptable while work remains, but completion requires an open, non-draft PR ready for review. Keep the PR
   title/body current and reuse an existing PR for this step instead of opening duplicates.
3. Save one handoff in the PR record: PR URL, base/head SHAs, contract changes, remaining adapters/removal owners,
   test commands/results, relevant hardware/toolchain identity, report links, and missing evidence. Record the
   implementation session ID and actual model/effort. Do not duplicate this into a separate reporting system.
4. Start a separate fresh reviewer session with AGENTS.md, both plans, the full diff/surrounding source, and raw
   verification evidence. Do not preload the implementer's conversation. Review the whole change independently;
   the per-PR focus is guidance, not a scope limit. Enforce the no-backward-compatibility rule.
5. Record findings with severity/locations, actual checks, gaps, base/head SHAs, and reviewer session/model/effort.
   Fix actionable findings in the implementation session and rerun affected checks. The independent reviewer
   checks revisions and integration changes, then records a verdict for the final base/head pair. Any subsequent
   change needs renewed review; a new reviewer session is optional for revisions within this same PR.
6. Watch the PR's required CI checks to completion. Inspect failing logs, fix the cause, push, and repeat until
   required checks pass on the latest head. Pending, failed, cancelled, or missing required checks are not green.
   Do not disable checks or weaken tests to obtain a green status. Respect the existing prohibition on editing
   `.github/` without a separate request; an external CI/access blocker must be reported, not disguised as success.
7. Finish only with an open, non-draft PR, green required CI for the final head, applicable local/evidence gates
   passed, and independent review reporting `no blocking findings` for the final base/head pair. Resolve
   actionable review findings. Required human approval is not supplied by an agent review; report any remaining
   merge requirement accurately. Missing required hardware evidence still means `verification incomplete`.
8. Verify the final PR state and preserve its URL, final SHA, check results, and independent review in the
   handoff. Only then mark the session goal complete with `update_goal` and report completion. Do not end at
   "PR opened" while CI is running. Keep the goal unfinished when requirements remain, following the goal
   tool's blocked-state rules where an external blocker persists. Leave merging to the user.

**Review and verification gates**

The following named gates are used throughout the sequence. Each PR runs the checks appropriate to its changes;
they are not reasons to repeat unchanged hardware suites unnecessarily.

| Gate | Required evidence |
|---|---|
| G1: repository | `./gradlew :koblas:check :koblas-bench:check lintDocs`; run the bundled-C configuration with `-Pkoblas.noSimd=true` where dispatch/native behavior changes. |
| G2: native ABI/build | JVM shared library and Kotlin/Native static archive compile/link on affected supported targets; ABI layout/export tests; library loads on baseline hardware. Cross-compilation alone is not execution evidence. |
| G3: numerical | New kernels agree with an independently callable scalar oracle using existing assertion conventions. Include no-read cases, special values, logical edges, permitted aliases, and untouched backing storage. |
| G4: exact ISA | Requested SME and SME2 implementations execute on supported hardware, with actual ID/layout recorded. Audit generated instructions and state preservation; use feature-restricted execution where available for the SME-only path. |
| G5: performance | Reproducible repository-harness comparisons with matching inputs/timing boundaries, raw and full-operation costs, and retained-packed cases where relevant. No inferred thresholds from instruction peak throughput. |
| G6: runtime | Allocation/lifetime checks, concurrency, per-thread execution constraints, and bounded JVM critical-call/safepoint behavior. |
| G7: independent review | Mandatory for every PR: separate fresh reviewer session, explicit model/effort, review of the final base/head pair, evidence-backed verdict, and resolution of findings before completion. |
| G8: open PR and green CI | Mandatory for every PR: pushed final commit, open non-draft GitHub PR, required checks passing on its latest head, and PR URL/SHA/check evidence in the handoff. |

G7 and G8 apply to every PR alongside its listed technical gates. Every applicable correctness gate must pass before
merge. A kernel without real-hardware execution evidence is
not a verified supported backend. A correct but uncalibrated kernel may remain exact-only. If target hardware
is unavailable, unrelated foundation work may land, but target execution/default activation remains outstanding.

Before pushing, run the full repository check as instructed by AGENTS.md. Keep HFactor implementation changes
out of this project; the full check may still rebuild it. Do not edit `.github/` without a separate request.
Match repository test naming/layout and keep individual JVM tests short by parameterizing bounded cases.

**Sequence and complete old-to-new mapping**

| PR | Included former steps | Suggested title | Essential prerequisites |
|---|---|---|---|
| 01 | W01 | `test: establish kernel transition baselines` | None |
| 02 | W02, W03, W04 | `feat: unify native builds and kernel capabilities` | 01 |
| 03 | W05, W06 | `feat: define kernel selection and matrix operands` | 02 |
| 04 | W07, W08 | `refactor: execute gemm through shared block kernels` | 03 |
| 05 | W09, W10, W11 | `feat: add SME and SME2 product kernels` | 02, 04 |
| 06 | W12, W13 | `feat: add SME and SME2 layout kernels` | 03, 05 |
| 07 | W14, W15 | `refactor: share structured matrix product execution` | 04–06 |
| 08 | W16, W17 | `refactor: decouple triangular blocks from product tiles` | 03, 04, 07 |
| 09 | W18, W19 | `feat: add SME and SME2 triangular kernels` | 05, 08 |
| 10 | W20, W21 | `feat: execute gemv through variable panel kernels` | 03–06 |
| 11 | W22, W23 | `feat: add symmetric and rank update block kernels` | 07, 10 |
| 12 | W24 | `perf: tune native batches and workspace lifetimes` | 04–11 |
| 13 | W25, W26 | `perf: calibrate and enable operation specific dispatch` | 01–12 |
| 14 | W27 | `refactor: finalize the kernel architecture migration` | 13 |

**PR 01 — Explicit benchmark cases and baselines**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `high`.

**Goal:** Preserve explicit benchmark recipes, numerical coverage, and reproducible baselines; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**Status:** landed in GitHub PR #540. The goal and requirements below document acceptance; do not start a duplicate PR.

**W01 — Establish semantic and benchmark baselines**

Transition: current behavior is implicit in scattered tests and fixed physical benchmark cases; after this PR,
the redesign has an explicit reference and comparable measurements.

- Inventory zero-alpha/beta, empty dimensions, robust norms, selected triangles, unit diagonals, strided views,
  permitted aliases, and the ordered numerical fallbacks. Record each contract's current oracle/test owner.
- Extend the harness's case descriptions to distinguish logical workload from backend physical layout without
  changing the timing meaning of existing rows. Identify changed workload/fixture semantics by explicit
  case/recipe and source SHA; do not mix incompatible reports or restore removed version-label fields. Add
  shapes that expose skinny GEMM, depth tails, and multi-RHS solves.
- Put the tile/packed settings from architecture section 13 explicitly in `koblas-bench/cases.txt` before
  capturing baselines: an explicit packed recipe fixing tile, left/right formats, groups/strides/padding/alignment,
  plus independently varied block/panel settings, variant constraints, and timing mode. Require the recipe and
  applicable independent fields even when today's defaults match; avoid duplicating constants it already fixes.
  Kotlin/vendor runners must consume them without consulting backend/tuning defaults to reconstruct the workload. Validate actual geometry/layout against the case; reject unsupported selections.
- Implement fixed-configuration and logical-workload comparison modes. Generate logical fixtures before
  packing, retain physical-work metadata, and include old tile-loop overhead when timing a logical block.
  Update Kotlin/vendor runners and comparator together. Preserve semantic identity across kernel renames;
  old reports without sufficient equivalence metadata require a fresh old-code baseline.
- Capture current scalar, JVM SIMD, JVM C, and Native baselines on available representative targets. Record
  missing targets explicitly. Do not make a performance test a timing assertion in ordinary unit tests.
- Add meaningful missing reference coverage; do not copy the current implementation into expected values.

Verification: G1, G3, G5 baseline capture. Exit: later reports can compare identical logical problems and explain
any fixture or timing-boundary change. Test missing-field rejection and show that changing backend/tuning
defaults cannot change a fixed case's workload. Test cross-layout comparisons and rejection of incompatible
fixture/timing/physical-work pairs. No production dispatch changes.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W01: oracle independence, no-read/alias coverage, explicit case-defined work, and honest before/after comparisons.

**PR 02 — Native builds, capability probe, and ordinary widths**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver one native build pipeline, a baseline-safe capability probe, and exact ordinary-width execution; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W02 — Extract and unify native builds**

Transition: header-compiled implementation becomes ordinary compiled native code shared by both runtimes.

- Move bodies from `koblas_kernels.h` and `koblas_packed_trsm.h` into implementation/internal files. Public
  cinterop headers become declarations; preserve behavior and current callable symbols for this transition.
- Build target-specific static archives for Kotlin/Native and shared libraries for JVM from the same sources.
  Update `koblas_kernels.def`, Gradle dependencies/inputs, resource checks, and exported-symbol verification.
- Track compiler/toolchain identity, target triple, flags, and all sources in build cache inputs. Handle cross
  targets explicitly and retain the existing supported scalar cross-target behavior.
- Keep current instruction selection intact until W04 makes variants explicit. Do not introduce SME here.

Verification: G1, G2, existing kernel conformance and a small G5 before/after sample. Exit: each affected target
links the correct architecture; no host archive leaks into another target. Old symbol forwarding is removed
after the last consumer migrates, no later than W27.

**W03 — Add the generic C probe and state model**

Transition: library availability becomes a versioned catalog of built, usable, and unavailable kernel choices.

- Implement `koblas_probe_v1` HOST/KERNEL/THREAD queries with struct sizing/version negotiation, stable IDs,
  feature records, type tuples, widths, matrix geometry, layouts, and reason codes. Document error behavior.
- Implement baseline feature discovery for current x86/Arm targets; distinguish hardware, build, OS support,
  and current execution context. Querying never changes permissions or executes an illegal-instruction trial.
- Describe process/thread/per-call state and test preparation outcomes with synthetic descriptors. Add a real
  preparation entry point only if a supported backend needs it; do not ship a dummy AMX/ACE permission service.
- Add JVM and Native probe bindings, using non-critical JVM calls for OS queries/preparation. Decode the same
  records on both runtimes; no independent Kotlin CPU-feature guesswork.
- Test synthetic ACE-without-FP64, AVX10-without-ACE, denied enablement, unknown feature records, and a future
  FP64 tile descriptor. Synthetic capability tests cannot authorize real unsupported instructions.

Verification: G1, G2 and descriptor/ABI tests. Exit: unsupported and unavailable are explainable; process,
thread, and per-call state requirements are distinct. The probe is useful before SME kernels exist.

**W04 — Make ordinary C variants and widths explicit**

Transition: hidden target cloning/source-vector width assumptions become individually identifiable kernels.

- Give baseline and existing ordinary SIMD implementations stable operation-specific IDs. Separate targeted
  objects or explicit specializations so the exact selected ISA can be verified.
- Expose typed vector/panel execution entry points resolving an ID once outside the loop. Wire JVM and Native
  calls through them. Preserve a scalar/non-accelerated competitor for every migrated contract.
- Describe machine register width, logical batch size, and unroll separately. Provide the specialization seam
  for future NEON/SSE2/AVX2/AVX-512/SVE/AVX10 choices without claiming unimplemented variants.
- Keep pure scalar Kotlin and JVM SIMD identities distinct from ordinary native SIMD. Add exact raw C mode
  that bypasses performance thresholds while preserving semantic early exits.

Verification: G1–G3; disassembly and G5 for affected ordinary variants. Exit: actual native ISA/width is
attributable and exact requests cannot resolve to a different width. Remove obsolete hidden clones here.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W02: cross-target archive selection, symbol/link parity, toolchain cache inputs, and baseline instruction safety.
- W03: ABI sizing/version negotiation, capability versus readiness, permission scopes, and rejection of ACE without FP64.
- W04: truthful ISA and width IDs, absence of hidden substitution, and baseline portability.

**PR 03 — Typed selection, operands, and scalar block contracts**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver small typed selection rules, validated windows/packed operands, and executable scalar block contracts; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W05 — Add immutable profiles and inspectable planning**

Transition: process-wide scattered crossovers become data belonging to a hardware/runtime execution policy.

- Add small typed profile tables under `koblas/src/tuning/` with deterministic validation. Separate legal native
  variants from scheduling, host crossovers, and algorithm choices. Kotlin owns policy and passes selected
  schedules to C; generate shared data only if a field actually has consumers in both languages.
- Implement exact/AUTO policies and inspectable selection functions in the existing engine where possible;
  do not require a new public planner/profile framework. Represent `Never`,
  `AlwaysEligible`, shape rules, and checked/saturating work estimates explicitly.
- Add operation/component diagnostics; an engine with mixed components cannot label every operation SME.
- Resolve properties/environment overrides once. Validate supported geometries and IDs; exact unavailable
  requests fail. Keep current proven behavior as transitional conservative profile data.
- Test scalar-to-C and SIMD-to-C as separate decisions, including cases where one stage never wins. Exercise
  synthetic x86 matrix descriptors so the planner is not organized around SME booleans.

Verification: G1, decision tests and warmed allocation checks. Exit: performance choice is separate from
capability, no startup autotuning is introduced, and newer ISA/greater width does not imply priority.
Existing operation families consume this policy as they migrate; residual old tuning readers belong to W27.

**W06 — Define windows, packed layouts, and executable block contracts**

Transition: raw panels and fixed microtiles become validated logical operands with an executable scalar contract.

- Define matrix/vector windows and retained packed operands: dimensions, offsets/strides, padding/alignment,
  layout ID/version, ownership, and baked transforms/scaling. Prefer unscaled reusable panels. Keep future type
  tuples extensible without adding quantization/scale-buffer machinery to Double operands.
- Implement scalar pack/unpack/transpose for general, symmetric, and triangular input. Validate caller-owned
  wrapping, sizes, permitted aliases, and retained-buffer lifetime independently of the initializing thread's SVL.
- Define direct, one-side-packed, both-side-packed, and retained product modes with alpha/beta, selected output,
  logical m/n/k, scratch requirements, and first/subsequent depth contributions. Implement the scalar reference.
- Keep ordered numerical fallbacks and failure-before-mutation explicit. Multiple kernels may share a declared
  layout; product packing dimensions must not dictate triangular solve order.
- Update in-repository callers/API dumps with the contract changes. Use temporary internal wrappers only for
  unmigrated consumers; remove them in the owning family PR, with the final audit in W27.

Verification: G1, G3, size/stride overflow, round trips, retained-layout mismatch, alpha/beta/no-read cases,
non-finite/extreme values, and untouched backing storage. Exit: layout and product contracts are exercised
together before accelerated implementations are added.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W05: immutable selection, checked shape arithmetic, separate scalar-C and SIMD-C policies, and truthful diagnostics.
- W06: ownership/layout validation, scalar-oracle independence, alpha/beta and no-read semantics, and failure before mutation.

**PR 04 — Ordinary GEMM blocks and shared matrix/view execution**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver ordinary C/JVM SIMD block kernels and one GEMM orchestration for owning matrices and views; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W07 — Implement ordinary C and JVM SIMD product blocks**

Transition: ordinary backends implement the final block API, proving it is not an SME-only abstraction.

- Implement direct and packed block products for ordinary C and JVM SIMD. Reuse valid arithmetic code where
  helpful, but put the microtile loops inside the selected backend's block execution.
- Add versioned C matrix execution bindings and block-level Kotlin/Native pinning. Add edge/masked stores and
  alpha/beta handling under the contract from W06.
- Establish a conservative finite work bound for JVM critical calls now. W24 optimizes it; no interim PR may
  pass an arbitrarily large whole BLAS operation through a critical downcall.
- Preserve the JVM inlining rule for helpers returning `DoubleVector`; benchmark allocation behavior.

Verification: G1–G3, G5 ordinary comparisons, G6 allocation/call-boundary smoke. Exit: every new product mode
has a non-SME execution path and exact diagnostics. No new portable ISA-specific traversal is introduced.

**W08 — Migrate GEMM and view execution to one planner**

Transition: `BuiltinBlas`/view defaults and per-tile Kotlin dispatch become shared operation-level scheduling.

- Route owning and view GEMM through one validation, alias, semantic-eligibility, and planning path.
- Select direct, one-side-packed, both-side-packed, or retained-packed execution before allocating/packing.
  Schedule cache blocks; backend code owns microtiles. Reuse workspace buffers across depth/row blocks.
- Use explicit scaling/writeback semantics and document internal panel packing versus a full view copy.
- Keep one portable GEMM orchestration for all exact and AUTO backend selections. Preserve independently
  callable reference arithmetic, not a second production portable implementation per ISA.
- Remove the superseded GEMM-only scheduler/edge adapters; shared helpers still used by unmigrated structured
  or triangular operations remain temporarily and have deletion owners in W14–W19.

Verification: G1–G3, G5 full GEMM versus W01 on ordinary backends, G6. Exit: matrices and views receive the
same planning opportunities; changing layout after packing is impossible. Regressions in the ordinary route
must be fixed before merge rather than hidden behind the future SME backend.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W07: C/JVM contract equivalence, edge handling, bounded calls, and allocation-free JVM vector helpers.
- W08: owning/view parity, pack amortization, alias handling, and removal of per-microtile foreign calls.

**PR 05 — SME execution boundaries and both FP64 product backends**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Verify isolated streaming boundaries and independently executable SME and SME2 FP64 GEMM kernels; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W09 — Establish isolated SME execution boundaries**

Transition: supported build systems gain verified SME and SME2 code-generation/link/runtime boundaries.

- Add separate SME-only and SME2 source targets, feature-tested compiler flags, target-correct helper linkage,
  and baseline-safe dispatch wrappers. Baseline-only builds remain valid; explicitly requested missing
  accelerator build components fail the build.
- Use ordinary pointer-based ABI entry points with compiler-managed streaming/ZA state. Keep scalable-vector
  values private to implementation functions with a compiler-supported ABI.
- Test state restoration, repeated calls, multiple threads, thread vector-length changes, and baseline library
  loading. Use private test helpers for state exercises; do not advertise a production kernel not yet built.
- Audit objects to ensure the SME-only target cannot acquire SME2 instructions and generic target flags cannot
  cause ordinary SVE instructions to run on an incompatible Apple target.

Verification: G1, G2, G4, G6. Exit: both runtimes can safely call an attributed implementation. The earlier
compile-only investigation is superseded by actual link and hardware evidence.

**W10 — Implement SME FP64 GEMM blocks**

Transition: the shared product planner gains its first matrix-accelerated implementation.

- Add SME-only `FMOPA` product kernels, multiple ZA accumulators, and tail-safe input/output handling.
  Honor the final block contract, including direct and supported packed modes.
- Interpret explicit packed strides and current SVL correctly. Specialized SVL restrictions are checked before
  output mutation; exact mismatch reports failure rather than quietly changing backend.
- Bind exact SME selection in JVM and Native. Exercise generic layouts through the scalar packer initially;
  the packing implementation is an independent plan component.
- Add candidate microtile shapes and record raw versus full-operation costs. Do not change AUTO defaults yet.

Verification: G1–G4, G5 initial SME results, G6. Exit: SME GEMM runs through the existing shared planner, including
owning/view and retained-packed paths, with no `smeGemm` portable fork.

**W11 — Implement SME2 FP64 GEMM blocks**

Transition: SME and SME2 become distinct, comparable product choices under the same contract.

- Implement SME2-specific data movement/output scheduling around the appropriate outer-product primitive.
  Use grouped instructions where they improve the measured schedule; a compiler flag alone is not an SME2
  implementation.
- Keep SME-only kernels independently available on SME2 hardware. Register exact IDs for actual schedules
  and legal layouts, including diagnostics when two paths share an arithmetic primitive.
- Test same-layout comparisons as well as each backend's native preferred geometry. Distinguish kernel gains
  from changes caused by packing or different physical shapes.

Verification: G1–G4, G5 SME-versus-SME2 raw and full GEMM, G6. Exit: both implementations pass identical logical
conformance cases and remain separately selectable. Neither receives a default solely because it is newer.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W09: streaming ABI attributes, ZA preservation, helper linking, per-context readiness, and baseline ISA isolation.
- W10: actual FP64 SME instructions, current SVL, predicated edges, epilogue semantics, and exact execution evidence.
- W11: distinct SME2 instruction schedules, feature gating, layout compatibility, and independently measured SME comparisons.

**PR 06 — SME and SME2 packing, unpacking, and transpose**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver independently measured SME and SME2 general/structured packing, unpacking, and transpose; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W12 — Implement SME layout kernels**

Transition: layout preparation becomes an accelerated component instead of an unconditional portable cost.

- Implement SME general packing, unpacking, and blocked transpose; add structural packing variants where their
  algorithm needs differ. Use existing explicit flags for mirroring, unit diagonals, and optional scaling.
- Produce exactly the declared final layouts. Test logical edges, padding, strided windows, bit-preserving
  copies where required, and never-reading ignored triangle/diagonal storage.
- Wire the layout planner for both high-level products and retained-panel APIs. Reuse source transformations
  rather than materializing a full matrix unnecessarily.

Verification: G1–G4, G5 layout-only and full GEMM, G6 scratch reuse. Exit: scalar and SME packers are compatible
when their layout ID matches, and an operation can mix a measured layout kernel with a compatible compute kernel.

**W13 — Implement SME2 layout kernels**

Transition: packing and final extraction gain independently measurable SME2 grouped-transfer schedules.

- Implement SME2 pack/unpack/transpose variants for the same contracts, including structural transforms used
  by later symmetric/triangular operations.
- Verify exact byte/layout compatibility with other producers/consumers; do not label an ISA-specific format
  generic unless every documented consumer supports it.
- Add independently configurable layout and compute choices, with end-to-end evidence that accounts for
  streaming transitions. Short padding clears retain an ordinary fill when appropriate.

Verification: G1–G4, G5 SME-versus-SME2 layout/full-operation comparisons, G6. Exit: packing and GEMM can be
tuned independently without multiplying portable algorithms.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W12: bit-preserving layout transforms, padding and tails, unit-diagonal/triangle no-read rules, and SVL-dependent layouts.
- W13: grouped SME2 moves, producer/consumer layout compatibility, tails, and complete packing costs.

**PR 07 — Structured products and fused SYR2K**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver shared SYMM/GEMMT/SYRK scheduling and the fused two-product SYR2K contract and backends; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W14 — Migrate symmetric and triangular-output products**

Transition: SYMM, GEMMT, and SYRK share the new product planner rather than old tile orchestration.

- Route each operation through backend-neutral scheduling, structural input descriptions, and selected-output
  masks. Share the GEMM block implementations; add specialized schedules only where the operation warrants one.
- Preserve triangle-only reads/writes and the existing semantic guards for rearranged arithmetic. Edge handling
  belongs to the block contract, not an unconditional Kotlin microtile scratch loop.
- Extend exact backend reports so coverage cannot silently skip a physical shape. Route relevant view entry
  points through the same shared logic.
- Remove the old structured-product scheduler when its last consumer moves; leave the SYR2K ordered semantic
  fallback until W15 establishes the new fused operation.

Verification: G1–G4, G5 all structured products and both triangles. Exit: SME/non-SME portable paths do not exist;
selection changes kernel/layout components only.

**W15 — Fuse SYR2K output accumulation**

Transition: two separate product/output passes become an explicitly contracted two-product update.

- Add a fused rank-2k block primitive with a scalar reference and ordinary C/JVM SIMD/SME/SME2 implementations.
  Both products share output traffic; beta is applied exactly once across products and depth blocks.
- Keep eligibility checks for scaling/reassociation and an ordered generic fallback. A declined accelerated
  call leaves the destination untouched.
- Route one shared SYR2K orchestration to the fused or ordinary product strategy. Delete obsolete duplicated
  scheduling after all callers move.

Verification: G1–G4, special-value and beta-once tests, G5 fused versus composed full operations. Exit: each
backend serves the same new primitive; the shared planner can retain composed execution when it wins.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W14: structured operand access, selected-triangle stores, fused scaling eligibility, and shared portable scheduling.
- W15: both-product accumulation, beta applied once, numerical fallback eligibility, and untouched output triangles.

**PR 08 — Shared triangular multiplication and solve scheduling**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver dependency-correct TRMM/TRSM scheduling and ordinary solve/update blocks with independent geometry; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W16 — Migrate TRMM block scheduling**

Transition: triangular multiplication becomes dependency-aware scheduling of final product/layout kernels.

- Implement one shared left/right/transposed triangular multiplication traversal. Choose safe in-place order
  or explicitly costed source staging; use the same product blocks for ordinary SIMD, SME, and SME2.
- Reuse packed source/triangle panels across calls, with structural metadata and selected input reads.
- Preserve zero-source, unit-diagonal, scaling, and alias behavior. A matrix-shaped problem does not authorize
  extra reads of the unstored triangle.
- Remove `packedTrmmCore` and its obsolete packing adapters after replacement is verified.

Verification: G1–G4, all side/uplo/transpose/diag combinations, G5 and G6. Exit: no TRMM scheduling dimension is
implicitly fixed by a backend's old 4-by-4 tile interface.

**W17 — Redesign TRSM orchestration and ordinary solve blocks**

Transition: solve order, RHS batching, and product geometry become independent.

- Define diagonal-solve and fused update-and-solve block contracts; implement their scalar oracle and ordinary
  C/JVM SIMD versions. Preserve division and zero-source/pivot semantics.
- Rewrite one shared TRSM traversal with independently tuned diagonal and RHS blocks. Reuse the product/layout
  infrastructure, and keep numerical fallback eligibility independent of a particular backend.
- Replace masks whose size was implicitly tied to a machine word where the final block design needs a wider
  representation; otherwise retain and validate a deliberate size bound. Never enlarge a block past its guard.
- Remove the old portable tile-shape dependency and migrate public/custom solve callers to explicit layouts.

Verification: G1–G3, G5 ordinary TRSM, G6. Exit: all flags and difficult numerical cases run without SME, and a
future accelerator only needs to implement the solve/update block contracts.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W16: in-place dependency order, side/transpose/unit-diagonal combinations, and snapshot lifetime.
- W17: solve order independent of product tiles, diagonal/RHS geometry, fused update contracts, and all triangular flags.

**PR 09 — SME and SME2 triangular block kernels**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver independently tested SME and SME2 diagonal-solve and fused update/solve kernels; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W18 — Add SME triangular kernels**

Transition: the shared solve traversal gains SME solve and fused product-subtraction/solve implementations.

- Implement `trsmTile`'s replacement and `gemmTrsmTile`'s replacement for SME-only execution. Parallelize across
  independent RHS entries; retain pivot dependency order.
- Keep residuals in registers where useful and include all transfers in benchmarks. A composition that uses
  ordinary SIMD for part of the solve reports that fact; exact kernel IDs identify the actual implementation.
- Cover zero-depth solve-only calls, short diagonal/RHS blocks, nonunit and unit diagonals, and padded storage.

Verification: G1–G4, G5 versus ordinary block solve and composed update+solve, G6. Exit: the portable TRSM code
does not change to accommodate SME; its block selection changes.

**W19 — Add SME2 triangular kernels**

Transition: SME2 grouped RHS updates become a separate solve/update candidate.

- Implement grouped residual extraction, updates, and substitution schedules appropriate to the final contracts.
  Keep FP64 division and required numerical behavior intact.
- Compare standalone diagonal solve, fused update/solve, and full TRSM, including shapes with few RHS entries.
- Remove final users of old `PackedKernels.trsmTile`/`gemmTrsmTile` adapters. Update packed benchmarks to operate
  on final contracts with exact layout metadata.

Verification: G1–G4, G5 SME/SME2 comparisons, G6. Exit: both architectures have complete triangular alternatives;
AUTO may later select an ordinary solve for shapes where it wins.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W18: SME solve dependencies, update/solve fusion, diagonal semantics, and truthful composed-kernel identities.
- W19: SME2 grouped triangular dependencies, all edge cases, state boundaries, and evidence against SME competitors.

**PR 10 — Variable panels and GEMV on ordinary, SME, and SME2 backends**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver variable-width panels, shared owning/view GEMV, both SME implementations, and migrated sparse callers; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W20 — Generalize panels and migrate GEMV**

Transition: fixed `dot4`/`axpy4` calls become variable-width panel execution with one GEMV traversal.

- Define and implement scalar, ordinary C, and JVM SIMD multi-dot/multi-column-update contracts. Include fused
  scaling/writeback and the correct zero-multiplier behavior.
- Rewrite GEMV to schedule row/reduction panels and hold outputs across more columns or rows. Unify owning
  and strided/view execution through validated windows.
- Route small cases through cheap in-runtime kernels. Remove the four-output API limitation rather than adding
  separate `dot8`, `dot16`, and SME-only portable loops.
- Migrate existing sparse panel consumers to this interface without changing their algorithms. Remove their
  four-column adapters here. Only unmigrated SYMV/rank consumers retain adapters, removed in W22 and W23.

Verification: G1–G3, G5 ordinary GEMV across both transposes, G6. Exit: the panel abstraction is useful without
SME and supports backend-specific execution group sizes without changing portable scheduling semantics.

**W21 — Add SME and SME2 GEMV/panel implementations**

Transition: both matrix backends execute the panel/GEMV contracts without new portable branches.

- Add SME panel schedules and distinct SME2 vector-group schedules for the applicable multi-dot and update
  operations. Register exact IDs; unsupported candidates are explicit rather than disguised as accelerated.
- Keep output/reduction state in the native region across a panel, rather than entering streaming mode once
  per old four-column leaf. Handle stride/edge behavior through the shared contracts.
- Preserve ordered coefficient evaluation where required, reduction semantics, and out/input alias rules.
- Measure skinny products and direct GEMV against ordinary kernels. Do not assume a ZA-based panel wins merely
  because its instruction throughput is higher.

Verification: G1–G4, G5 raw panels and full GEMV, G6. Exit: SME and SME2 panel alternatives are independently
testable. If the native source diff becomes too large, split this PR by ISA while keeping the same W20 contract.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W20: variable panel widths, transpose/stride paths, accumulation order, and owning/view parity.
- W21: SME and SME2 panel identity, streaming amortization, reduction tails, and full GEMV costs.

**PR 11 — Symmetric vector products and direct rank updates**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Deliver shared SYMV and GER/SYR/SYR2 block execution with numerical guards and ordinary/SME/SME2 kernels; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W22 — Replace four-column SYMV with symmetric blocks**

Transition: one symmetric block schedule replaces hard-coded regrouping and duplicated portable work.

- Define a coupled off-diagonal block update contributing both a block product and its transpose contribution;
  diagonal blocks read only the selected triangle.
- Implement the scalar oracle and ordinary/SME/SME2 block kernels; share the same portable SYMV traversal.
  Reuse components from W20 and W21 without reintroducing many foreign calls.
- Rework the numerical eligibility checks for the actual new schedule. Use preflight checks or scratch-and-
  commit when required; never fall back after mutating y and apply the contribution again.
- Remove the old SYMV four-column scheduler and its temporary panel adapters.

Verification: G1–G4, intermediate-overflow/cancellation cases and both triangles, G5/G6. Exit: one generic
ordered fallback and one shared blocked strategy remain; neither is named or branched by ISA.

**W23 — Add direct matrix rank-update blocks**

Transition: GER/SYR/SYR2 can update whole logical blocks instead of issuing one AXPY per column.

- Define rank-one/rank-two block contracts and implement scalar, ordinary C/JVM SIMD, SME, and SME2 candidates.
  Reuse product components where profitable while accounting for the small reduction depth.
- Route each public operation through a shared planner/traversal. Preserve raw zero-source skips, zero-product
  evaluation where required, selected-triangle writes, and aliases.
- Compare direct outer-product/grouped schedules with bandwidth-efficient ordinary vector updates. Advertise
  kernel availability separately from AUTO eligibility.
- Remove migrated column-update adapters; retain only genuinely useful general panel contracts.

Verification: G1–G4, G5 full rank updates with hot/cold destinations, G6. Exit: Level 2 rank updates use the same
backend-neutral matrix layer and do not impose GEMM packing when it cannot pay off.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W22: coupled symmetric updates, overflow preflight, no partial mutation before fallback, and numerical edge cases.
- W23: rank-update alpha semantics, skipped-zero behavior, selected triangles, aliases, and fused epilogue eligibility.

**PR 12 — Bounded native calls and workspace tuning**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Measure bounded native-call strategies and add an alternative storage path only where justified; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W24 — Optimize native batches and workspace lifetimes**

Transition: conservative safe call boundaries become measured per-runtime execution strategies.

- Tune the amount of work per block/panel call without changing the shared mathematical traversal. Split long
  reductions or depth blocks correctly and preserve first-contribution beta behavior.
- Measure whether bounded heap-array calls meet throughput/latency needs. Add a native-memory alternative
  only if justified, including staging, cleanup, reuse, ownership, and memory-budget costs. A second storage
  strategy is not required when the simpler route wins.
- Keep heap-backed critical calls bounded; no retained heap pointer or live ZA state crosses a Kotlin return.
  Pin Native arrays once per useful call. Keep workspaces independent across concurrent operations.
- Verify preparation/OS queries occur outside critical regions and that current-context incompatibility is
  handled before output mutation, not by retrying partially completed work.

Verification: G1–G4 for changed paths, G5 full-operation and reused-memory costs, G6 including safepoint/GC and
concurrent workloads. Exit: larger batches are justified by throughput and latency, not only by reduced call count.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W24: heap/native pointer lifetime, safepoints, concurrent workspaces, bounded calls, and beta across split reductions.

**PR 13 — Calibration, vector evaluation, and measured AUTO defaults**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Calibrate the completed kernel catalog, evaluate remaining vector candidates, and activate measured AUTO profiles; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W25 — Calibrate profiles and evaluate remaining vector candidates**

Transition: individual kernel experiments become a repeatable process that produces checked tuning profiles.

- Extend the existing harness/report tools with candidate sweeps for legal widths, microtiles, packing groups,
  cache blocks, triangular/RHS blocks, direct/packed choices, and call limits. Use the established compatibility
  checks; do not hand-edit generated benchmark data.
- Reuse existing vendor comparisons. Add an Accelerate binding only as a justified follow-up if current
  coverage cannot answer a target comparison; it is not a completion gate for this sequence.
- Include existing dot/sum/ssqd/asum/nrm2 and vector mutations across lengths/strides. Evaluate additional
  ordinary widths or streaming/grouped leaves only for plausible gaps; preserve robust norms and zero semantics.
  Negative results need a report, not a shipping kernel. Remove stale unconditional dispatch assumptions.
- Emit profile candidates with source reports, runtime/compiler identity, layout/kernel IDs, and workload
  provenance using source SHAs and explicit case/recipe records. Validate typed tables and reuse existing
  report tooling; add import/generation only to remove actual duplicated maintenance. Check cross-language
  agreement only for shared data. Do not add a general profile/configuration language.
- Measure scalar-to-C and JVM-SIMD-to-C independently. Include raw kernel, full-operation, and retained-packed
  boundaries. Validate candidate rules on held-out shapes and repeated runs.

Verification: G1, harness tests, G5 complete calibration runs. Exit: thresholds and schedule choices are
reproducible; unavailable hardware produces no invented calibrated profile. Calibration tools are not executed
at application startup.

**W26 — Activate measured operation-specific defaults**

Transition: the default engine's old whole-backend precedence becomes the final mixed execution policy.

- Check in measured profiles for the verified hardware/runtime combinations, including SME and SME2 exact
  competitors. Separate scalar-C, SIMD-C, native-C, and algorithm/packing thresholds.
- Enable AUTO choices per operation and shape. Small operations stay in-runtime when appropriate; large GEMM
  can use a matrix backend even when the JVM Vector API is enabled. Newer ISA need not win every operation.
- Restrict retained-packed execution to compatible layouts or an explicitly costed repack. Select layout before
  packing for ordinary calls. Never pay an extra hidden scalar crossover inside exact native execution.
- Keep unknown or uncalibrated hardware on conservative correct profiles. Exact selection remains available
  for verified implementations, and diagnostics explain fallback or disabled choices.
- Exercise synthetic ACE/AMX/AVX10 eligibility again: a future ISA must fit the planner, while ACE without FP64
  must remain ineligible for the Double arithmetic path. No speculative ACE instructions are added.

Verification: G1–G6, ordinary-hardware regression runs and held-out AUTO-versus-exact comparisons. Exit: enabled
choices are supported by end-to-end evidence; the presence/absence of the JVM vector module/native library is
fully covered, and no process-global provider mutation is introduced.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W25: timing equivalence, robust vector semantics, deterministic profile import, provenance, and held-out validation.
- W26: operation-specific defaults, independent JVM crossover rules, unknown-host behavior, and end-to-end regression evidence.

**PR 14 — Remove adapters and verify the release**

**Implementation:** fresh session; model `gpt-6-astra`; reasoning `xhigh`.

**Goal:** Remove all transitional seams and verify final API, packaging, numerical, hardware, and performance evidence; OPEN the GitHub PR,
pass all applicable gates and independent review, and get required CI green on the final reviewed head.
Leave it open and ready for review without merging; mark the session goal complete only then.

**W27 — Remove transitional seams and verify the end state**

Transition: any remaining bridge to the old architecture is removed; all callers use the final contracts.

- Remove obsolete legacy C symbols, fixed-shape/four-column adapters, engine-wide precedence, old tuning
  readers, and superseded packing/tile APIs. Keep intentional scalar references and generic numerical fallbacks.
- Finish source-level migration documentation for low-level/custom packed callers and config keys, with examples
  of exact selection, AUTO diagnostics, retained operands, and workspace ownership. Confirm all API dumps/KDoc.
- Audit every owning/view Level 2/3 entry point and every benchmark identity. Verify there is one shared
  portable orchestration per operation and no hidden SME/non-SME copies or scalar-only view bypasses.
- Package and inspect final target artifacts, their compiled catalog, and helper dependencies. Record verified
  hardware/toolchain coverage, target limitations, and final profile/report provenance.
- Run the complete release verification matrix and full repository check before pushing. If a required platform
  or operation lacks evidence, record it as unfinished support rather than marking the entire plan complete.

Verification: G1–G6 and the final checklist below. Exit: the architecture plan is implemented, measured defaults
are enabled where verified, and future width/accelerator additions do not require new portable BLAS families.

**Final independent review (G7):** fresh separate session; model `gpt-6-astra`; reasoning `xhigh`.

- W27: complete adapter removal, public/view entry-point coverage, packaged catalog accuracy, and final hardware evidence.

**Transition cleanup ownership**

| Temporary component | Earliest relevant work | Removal owner |
|---|---|---|
| Old native symbol forwarding | PR 02 / W02 | Migrate callers directly within PR 02 where possible; later owning PRs remove remaining uses; PR 14 audits none remain. |
| Hidden ISA clones | PR 02 / W02 | PR 02 / W04, in the same PR. |
| Old tuning readers for unmigrated operations | PR 03 / W05 | Each family migration; PR 14 verifies none remain. |
| Old raw-panel wrapping | PR 03 / W06 | PRs 04 and 07–09; PR 14 covers residual public/harness callers. |
| Old product/tile output helpers | PR 04 / W08 | PRs 07–09 after structured/triangular consumers migrate. |
| Four-column panel adapters | PR 10 / W20 | Sparse callers migrate in PR 10; PR 11 / W22–W23 removes SYMV/rank-update consumers. |
| Conservative operation preferences | PRs 03–12 | PR 13 / W26 replaces only those with verified measured policies; conservative unknown-host behavior remains intentional. |

**Final acceptance checklist**

- Every implementation PR began in a fresh session and ended with an independent fresh-session review of its
  final base/head pair. Model/effort, evidence, findings, resolutions, and handoffs are recorded for all PRs.
- Every session created an explicit goal and delivered an open, non-draft GitHub PR with required CI green on
  the final reviewed head before marking that goal complete. Local-only changes and pending CI do not qualify.
- Both SME-only and SME2 product, triangular, panel, and layout implementations execute independently through
  the shared operation contracts; exact reports identify any deliberately composed components.
- The generic C probe describes actual implementations, numerical types, ordinary widths, packed layouts,
  accelerator geometry, and readiness. Its ABI and synthetic future-ISA tests pass on non-Arm hosts too.
- GEMM, structured products, TRMM/TRSM, GEMV/SYMV, and rank updates use one backend-neutral portable
  orchestration per operation, with generic strategies and independently callable scalar reference semantics.
- Owning matrices and views receive the same applicable acceleration; packed data carries its own layout and
  remains safe across threads/context changes, with explicit rejection/repack rules where needed.
- Profile data separately controls native schedules, algorithms, and host crossovers. Defaults are measured;
  ordinary SIMD widths and future AMD/Intel accelerators can be added without changing the portable architecture.
- JVM scalar-to-C and JVM-SIMD-to-C decisions are separate, shape-aware where needed, and compared against
  actual warmed JVM behavior. Neither ISA names nor instruction-throughput figures serve as thresholds.
- All supported numerical/alias/no-read contracts pass; allocations, native lifetimes, threading, and JVM call
  latency have evidence. Faster raw kernels with slower full operations do not become AUTO defaults.
- ACE extensibility is tested without claiming published low-precision arithmetic as an FP64 backend; adding
  a future element family remains an explicit API project.
- No transitional adapters or duplicated portable ISA families remain. The repository builds and checks pass;
  final support claims are limited to actually validated targets and modes.
- No obsolete API, ABI, packed format, configuration key, or dispatch path survives solely for backward
  compatibility. All in-repository consumers, API dumps, and documentation reflect the final design.
