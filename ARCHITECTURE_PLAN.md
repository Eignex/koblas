# Architecture roadmap for expert control

## Purpose

Evolve Koblas for numerical-optimization experts who need deterministic algorithm selection, observable
performance behavior, explicit resource ownership, and tight allocation control. Preserve the existing
reference semantics, column-major and CSC storage, destination-passing APIs, alias guarantees, and Kotlin
Multiplatform support.

Maven Central installation and publication choices are out of scope.

## Architectural principles

1. Do not silently change the numerical algorithm because an unrelated backend artifact is present.
2. Make the exact backend, fallback, threshold, and relevant numerical policy observable.
3. Keep convenience APIs, but make explicit immutable contexts the preferred expert API.
4. Compare backend priorities only among providers of the same semantic role.
5. Make native ownership deterministic; cleaners are leak protection, not the primary lifecycle.
6. State allocation guarantees precisely and test them per backend.
7. Preserve portable reference behavior as the semantic oracle.
8. Introduce changes additively where possible and give migrations for intentional API replacements.

## Primary pain points

### Sparse providers compete despite serving different roles

`F64SparseDecompositions` puts general LU, repeated-pattern LU, symmetric factorizations, and
basis-oriented implementations behind one ranked seam. Adding BASICLU or KLU can therefore change what an
ordinary `SparseMatrix.lu()` means. Equal-priority host discovery can also select a different solver by
registration order on JVM and Kotlin/Native.

Target design:

- Separate semantic roles such as general sparse LU, repeated-pattern LU, sparse Cholesky, sparse LDL, basis
  factorization, and basis solver.
- Use priority only to choose between implementations of the same role.
- Give general sparse LU a stable documented default.
- Require an explicit strategy or provider for specialized workflows.
- Represent optional behavior with typed capability interfaces rather than concrete implementation casts.

### Dispatch is global and opaque at operation level

Free functions and operators use a process-wide context. A selected host half may still execute a portable
implementation because of a size gate, missing routine, or unavailable library. Slot-level acceleration
checks cannot describe the route of one operation.

Target design:

- Provide an immutable context builder/resolver seeded with reference defaults.
- Support `AUTO`, `NATIVE_ONLY`, and `PORTABLE_ONLY` dispatch policies.
- Support allow, warn, and throw fallback policies.
- Expose structured capabilities and a route decision for an operation and problem shape.
- Report provider version, ABI, threading, threshold, and fallback reason where available.
- Retain the process-wide registry and free functions as a convenience layer.

### Numerical controls differ between host and bundled providers

Host configurations expose useful controls, while bundled constructors hide many of them. Library location
and numerical policy are conflated.

Target design:

- Split loading/location settings from algorithm and execution options.
- Give host and bundled providers the same numerical option types.
- Preserve controls for OpenBLAS threads and dispatch thresholds, KLU ordering/BTF/pivot/storage policy, and
  UMFPACK scaling/refinement/pivot policy.
- Make process-wide side effects such as OpenBLAS thread changes explicit in capabilities and documentation.

### Native factors have nondeterministic lifetime

Native sparse factors are reclaimed primarily by cleaners, and the common factorization interface has no
deterministic close operation.

Target design:

- Make resource-owning factors deterministically closeable with idempotent `close()` and `use` support.
- Keep portable factors cheap and compatible, with a no-op close where useful.
- Keep cleaners as a last-resort leak guard.
- Specify behavior after close and test close/use/race safety.

### Dense containers do not support real views

Dense matrices require an exact contiguous backing and vectors have no offset or stride. Rows, columns,
panels, and solver work-array regions therefore require copies.

Target design:

- Add matrix views with offset, shape, and leading dimension.
- Add vector views with offset, length, and stride.
- Keep optimized contiguous fast paths.
- Define borrowed and owned buffer semantics explicitly.
- Consider separating immutable `CscPattern` from mutable numeric values so symbolic analyses and several
  matrices can safely share a pattern.

### Allocation guarantees are incomplete

`Workspace` pools exact-size `DoubleArray` buffers only. It cannot supply integer, native, FFM, or
backend-specific scratch. Some native solve paths allocate even when the common contract suggests otherwise.

Target design:

- Define allocation guarantees as no size-dependent managed allocation, no managed allocation, or no
  managed/native allocation.
- Add typed scratch and backend plan support.
- Add an optional strict policy that fails when an operation cannot honor its requested guarantee.
- Test hot-path contracts for every native backend as well as the portable implementation.

### Sparse reuse and diagnostics are too implementation-specific

KLU refactorization requires narrowing to a concrete class. Common sparse factors support only single-vector
solves and expose minimal diagnostics.

Target design:

- Model symbolic analysis, numeric factorization, and same-pattern refactorization explicitly.
- Add multiple-right-hand-side sparse solves into caller-owned dense destinations.
- Add a common report with applicable fields such as permutations, fill ratio, pivot range, ordering,
  scaling, refinement, memory use, inertia, and backend identity.
- Keep backend-specific detail extensible without forcing every backend to fabricate unsupported statistics.

### Expert documentation is incomplete

The package documentation refers to missing README sections for element types and BLAS coverage. There is no
single capability matrix or complete contract for routing, allocation, ownership, and concurrency.

Target design:

- Add routine-by-backend-by-target coverage.
- Document routing thresholds and fallback behavior.
- Document aliasing, allocation, lifecycle, and thread-safety guarantees.
- Add decision guides for general sparse LU, repeated-pattern solves, interior-point systems, and simplex.
- Replace clusters of Boolean BLAS flags with typed `Uplo`, `Transpose`, `Diag`, and `Side` values while
  retaining migration overloads where practical.

## Proposed stacked PR sequence

Each PR should compile and pass verification independently. Later PRs may depend on earlier ones, but no PR
should mix mechanical migration with an unrelated numerical implementation.

### PR 1: Add structured backend capabilities and routing diagnostics

Status: implemented and verified on `codex/backend-routing-diagnostics`; PR publication in progress.

- Introduce public semantic backend-role identifiers outside `internal` packages.
- Add structured context/backend status rather than relying on `koblasInfo` strings.
- Represent selected provider, availability, portability, version/ABI when known, thresholds, and fallback
  reason.
- Add operation-level route inspection for representative dense and sparse calls.
- Preserve existing selection behavior in this PR.
- Document and test the status model.

Acceptance criteria:

- A caller can determine whether a representative operation and shape will execute natively before running it.
- Existing APIs retain their behavior.
- `./gradlew check lintDocs --rerun-tasks` passes.

### PR 2: Add explicit immutable context resolution and fallback policies

- Add a context builder/resolver seeded from portable defaults.
- Add automatic, native-only, and portable-only policies plus explicit fallback handling.
- Make expert examples use a held context directly.
- Keep process-wide registration and free functions as convenience APIs.
- Test two differently configured contexts in the same process without global mutation.

Acceptance criteria:

- Independent solver instances can use different policies concurrently.
- Native-only mode cannot silently run a portable implementation.
- Existing global calls remain source compatible.

### PR 3: Split sparse selection into semantic roles

- Separate general LU, repeated-pattern LU, symmetric factorizations, and basis roles in context resolution.
- Establish and document a stable general-LU default.
- Stop specialized providers from changing unrelated sparse operations merely by being present.
- Replace concrete casts with typed capability lookup.
- Add cross-target selection tests with multiple available providers.

Acceptance criteria:

- Adding BASICLU does not change general sparse LU selection.
- JVM and Kotlin/Native resolve the same policy to the same semantic provider role.
- KLU repeated-pattern and basis workflows remain explicitly reachable.

### PR 4: Unify host and bundled numerical options

- Separate library location/loading from numerical and execution policy.
- Make bundled OpenBLAS, KLU, UMFPACK, BASICLU, and HFactor accept the applicable common option objects.
- Preserve existing constructors with delegating overloads or a documented migration.
- Report effective options through structured diagnostics.

Acceptance criteria:

- Every numerical option exposed by a host provider is also configurable for its bundled counterpart when
  the bundled library supports it.
- Tests verify effective option propagation rather than constructor storage alone.

### PR 5: Add deterministic native factor lifecycle

- Add an idempotent close contract to sparse factorization resources.
- Implement explicit release for JVM and Kotlin/Native KLU, UMFPACK, CHOLMOD, BASICLU, and HFactor resources.
- Retain cleaner fallback without double-free risk.
- Define and test operations after close.

Acceptance criteria:

- `use` releases native resources promptly on success and exception.
- Repeated close is safe.
- Cleaner fallback and explicit close cannot double free.
- Host conformance tests cover lifecycle behavior.

### PR 6: Introduce symbolic/numeric sparse factorization capabilities

- Add explicit pattern analysis, numeric factorization, and compatible-value refactorization abstractions.
- Implement KLU first without forcing unsupported providers into fake reuse.
- Consider UMFPACK/CHOLMOD analysis reuse in follow-up commits within this PR only if the abstraction remains
  coherent and the stack stays reviewable.
- Use pattern identity/versioning where possible instead of repeated `O(nnz)` structural comparisons.

Acceptance criteria:

- A same-pattern loop can analyze once and refactor values without concrete backend casts.
- Pattern incompatibility is reported explicitly.
- Resource ownership of analysis and numeric factors is unambiguous.

### PR 7: Add strided dense views and explicit buffer ownership

- Add offset/stride vector views and offset/leading-dimension matrix views.
- Adapt reference kernels first, followed by native BLAS/LAPACK paths supported by their ABI.
- Keep contiguous containers and fast paths intact.
- Add overlap and aliasing tests for panels, rows, columns, and shared buffers.
- Introduce borrowed/owned terminology and migration documentation.

Acceptance criteria:

- Matrix panels and array slices can be used without copying.
- Existing contiguous benchmarks do not regress materially.
- Alias behavior is documented and tested.

### PR 8: Strengthen workspace and allocation contracts

- Add typed managed and native scratch or operation-specific reusable plans.
- Add strict allocation policies and structured allocation capabilities.
- Remove avoidable per-solve sparse-native scratch allocation.
- Expand allocation tests across native sparse backends.

Acceptance criteria:

- Each hot-path API states and tests its allocation guarantee.
- Strict mode reports an unsupported guarantee before silently allocating.
- Repeated supported solves/refactors honor their declared contract.

### PR 9: Add sparse block solves and factorization reports

- Add multiple-right-hand-side sparse solve and solve-into APIs.
- Add a common extensible factorization report.
- Populate backend-specific fields where reliable.
- Add reference-conformance and alias tests.

Acceptance criteria:

- Sparse block solves avoid one foreign call per right-hand side where the backend supports a block solve.
- Reports distinguish unavailable statistics from valid zero values.

### PR 10: Complete the expert-facing API and documentation

- Add the capability/coverage matrix and workflow guides.
- Repair stale README/package-doc references.
- Document threading, lifecycle, routing, ownership, and allocation contracts.
- Add typed BLAS flags with migration overloads.
- Include complete examples using explicit contexts, strict dispatch, closeable factors, symbolic reuse, and
  workspaces.

Acceptance criteria:

- A numerical expert can select a backend and algorithm, predict routing and allocation, manage resources,
  and implement a repeated solve without reading internal source.
- `./gradlew check lintDocs --rerun-tasks` passes.

## Working rules for the long session

- Re-read this file before planning each PR and update it when an accepted design changes the remaining stack.
- Inspect the current branch and prior stack commits before starting the next PR.
- Keep each commit single-line Conventional Commits without scopes, bodies, or trailers.
- Run focused tests while developing and `./gradlew check lintDocs --rerun-tasks` before completing each PR.
- Use `-Pkoblas.hostTests=true` when a PR changes native bindings; add `-Pkoblas.noSimd=true` when thresholds,
  level-2 bindings, or small-size dispatch are involved.
- Preserve portable reference semantics and compare backends through existing reference-conformance helpers.
- Do not edit `.github/` unless explicitly requested.
- Do not begin a later PR while an earlier PR has unresolved API or correctness questions that would change it.
- Keep this roadmap current: mark completed PRs, record deviations and rationale, and split a PR if its review
  surface becomes too broad.

## Definition of done

The roadmap is complete when an expert user can, without process-global mutation:

1. Construct an immutable execution context selecting semantic algorithms and exact providers.
2. Require native or portable execution and inspect the route for a particular operation and shape.
3. Apply the same numerical controls to host and bundled providers.
4. Deterministically release native analyses and factors.
5. Reuse a sparse symbolic analysis and refactor values through typed capabilities.
6. Operate on strided and submatrix views without copies where the backend supports them.
7. Request and verify a precise allocation guarantee.
8. Solve sparse systems with one or multiple right-hand sides into caller-owned destinations.
9. Obtain useful, extensible factorization and backend diagnostics.
10. Find all of these contracts and workflows in the public documentation.
