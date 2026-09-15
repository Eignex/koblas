# K1 contracts

The decisions K2, K3, K4 and the consumer migrations build against, fixed here so that independent tracks
cannot settle them differently. Recorded as the plan's K1 contract step requires; the dense binding contract is
additionally expressed in code, and this file says where.

## Dense operation and overload matrix

Expressed in code as `com.eignex.koblas.vendor.VendorOperation`, one entry per bound CBLAS symbol. Twenty-three
double-precision operations: Level 1 `dot`, `axpy`, `scal`, `copy`, `nrm2`, `asum`, `iamax`, `swap`, `rot`;
Level 2 `gemv`, `symv`, `ger`, `syr`, `syr2`, `trsv`, `trmv`; Level 3 `gemm`, `symm`, `syrk`, `syr2k`, `trmm`,
`trsm`, `gemmt`.

Overloads take views (`MatrixWindow`, `VectorWindow`) and nothing else; see *Owning and view signatures* below.

### Nonstandard operations

`gemmt` is the one operation outside the standard. oneMKL, AOCL and OpenBLAS export `cblas_dgemmt`; Accelerate
does not, and ArmPL is not assumed to. It is bound directly where exported and composed from `gemm` plus a
triangle copy where it is not, and the route of a call says which happened rather than leaving the operation
name to imply the first. The composed path keeps the direct one's contract: the opposite triangle of `C` is
neither read nor written, and `beta == 0` does not read the destination.

No other nonstandard entry point is bound. `sum`, `ssqd`, the fused panel kernels (`dot4`, `axpy4`, `dot-axpy`,
`axpy-arithmetic`), `rotm` and `rotmg` stay Kotlin implementations. A vendor arm reports them unsupported with a
reason rather than timing a composition of CBLAS calls under their name.

### Numerical contracts vendor BLAS cannot provide

`iamax` over a negatively strided window. BLAS returns zero for a non-positive increment rather than walking a
vector backwards, so the binding passes the increment's magnitude and reaches the same entries in the opposite
order. For `nrm2` and `asum` the order does not matter. For `iamax` it decides ties: among entries of equal
largest magnitude a forward window reports the first and a backward one reports the last. This is a documented
contract difference, not an accident, because no vendor flag expresses the forward-scan choice.

Solver-specific checked arithmetic is not a dense concern and stays with Klause; see below.

## Loading, ABI and threading

- **Vendor set.** Closed: Accelerate, oneMKL, AOCL, ArmPL, plus OpenBLAS as a benchmark reference that
  production selection never returns.
- **Order.** Operating system and architecture before CPU manufacturer. macOS takes Accelerate; Linux ARM64
  takes ArmPL whatever the manufacturer; Linux x86-64 takes AOCL then oneMKL on AMD, oneMKL then AOCL
  otherwise. An unsupported host selects nothing.
- **ABI.** Double precision, column-major, 32-bit BLAS integers matching Kotlin `Int`, carried by the
  unsuffixed CBLAS symbols. Verified two ways at load: a build advertising 64-bit integers in its own
  configuration string is rejected, and a known-answer probe must return exact expected values. The integer
  width cannot be settled by calling the library, because a 32-bit argument arrives in a register whose upper
  half is zeroed and an ILP64 routine reading 64 bits sees the same small value.
- **Precedence.** Installed libraries win; a bundled payload is a fallback. An installed library is the one the
  operator chose, and preferring a payload over it fails quietly, with everything still computing and only the
  version and performance not what was meant.
- **Bundle layout.** `com/eignex/koblas/vendor/<platform>/<file>`, with platform one of `linux-x86_64`,
  `linux-arm64`, `macosx-arm64`. One string addresses a JVM classpath resource and a file laid out beside a
  Native executable, so the optional module K4 publishes has one layout rather than two. Accelerate and
  OpenBLAS are never bundled.
- **Threading.** One compute thread per invocation, established at load before any arithmetic, including before
  the ABI probe, because oneMKL resolves its threading layer on the first call that reaches it. A library that
  still reports more than one thread never becomes a binding. Nothing is configurable and no thread count is
  reachable through any API, property or override. Bindings stay reentrant and concurrently callable; the limit
  is per invocation, not a process-wide lock.
- **Accelerate.** The one vendor whose thread count cannot be read back. Held to one thread through
  `VECLIB_MAXIMUM_THREADS`, set in-process before Accelerate performs arithmetic and overwriting any inherited
  value, and reported as unconfirmed rather than claimed as checked. Unverified on macOS hardware.
- **Missing backend.** Containers, Level 1 and the generic primitives do not go through vendor selection and
  keep working where it resolves nothing. Only accelerator-dependent calls fail, and they fail clearly.

## Owning and view signatures

The vendor binding takes views only. `MatrixWindow` and `VectorWindow` already carry offset, both strides,
structure, transpose and diagonal offset, which is the whole of what a BLAS call needs, and an owning container
converts to a view without copying. Two signatures per operation would mean two validation paths, and the plan
requires one.

The owning containers stay the public surface callers use. K3 owns the seam that turns `DenseMatrix` and
`DenseVector` into windows on the way to a binding, as part of routing the dense engine through it, and that
seam is the single validation and adaptation path the plan asks for. K1 fixes the view side of it and adds no
owning overloads that K3 would have to reconcile.

### Ownership rules

These hold for every binding call and are what the K3 seam has to preserve when it converts an owning container.

- **Storage is borrowed.** A window does not own its array and does not extend its life. The caller keeps the
  backing array reachable for the duration of the call; nothing in the binding retains it afterwards, and no
  foreign pointer outlives the call that made it.
- **No retained native copies.** An operand copied into native memory for a JVM call lives in that call's arena
  and is freed with it. Caching a copy of a mutable array across calls would be a correctness bug, not an
  optimization, and is not done. Any future retained native storage is a separate API decision.
- **Outputs are written through the window given.** A call writes only the entries its destination window
  addresses, plus, for a directly addressed window, the padding its own leading dimension already spanned,
  which is carried across a transfer unchanged. Storage outside the window is left as the caller left it.
- **Aliasing is the caller's to declare.** Distinct windows may share one backing array where the operation's
  own contract permits it, as `trsm` and `trmm` do for their in-place destination. Where an operation forbids
  overlap between an input and its destination, the binding requires it rather than discovering it late.
- **Concurrency is the caller's to synchronize.** A binding is immutable and concurrently callable, and each
  in-flight call owns its scratch. Two concurrent calls may read the same storage; a write overlapping another
  call's read or write is the caller's to order.
- **Structure is declared, not inferred.** A window's `structure` says which triangle is stored and whether the
  diagonal is implicit. The binding passes that through as flags and never reads an entry the structure says is
  not there, so an implicit unit diagonal may hold anything.

## Common sparse primitives

The surface K2 implements and Klause consumes. Every primitive is usable from `commonMain` with no platform
code in the consumer, operates on caller-owned buffers, and allocates nothing per call.

| Group | Primitives |
|---|---|
| CSC construction and iteration | column pointer and row index access, per-column value ranges, structural iteration without materializing a column |
| Triplet compression | sort and combine coordinate triples into validated CSC, summing duplicates |
| Indexed Level 1 | indexed dot against a dense vector, indexed dot against a second sparse support, indexed AXPY, indexed norm |
| Dense Level 1 | dense AXPY over a contiguous or strided destination |
| Scatter and gather | scatter a support into a dense accumulator, gather a dense accumulator into a support, gather-and-clear |
| Touched-index bookkeeping | record touched positions, compact them, clear only what was touched |
| Masked reduction | absolute maximum over an active column restricted by a mask |
| Scratch | caller-owned double and int buffers lent and returned without allocation |

### Semantics that must be stated, not inferred

- **Ordering.** Row indices ascend within a column. A gather emits in the order its touched-index list records,
  which is the order positions were first touched and not necessarily ascending; a caller needing ascending
  order compacts first. Validation rejects a CSC whose columns are not ascending.
- **Repeated indices.** Rejected on construction from CSC. Triplet compression sums duplicates instead, which
  is the one place a repeat is meaningful.
- **Exact-zero compaction.** A stored exact zero is structural and is preserved; compaction removes only what a
  caller asks it to remove. Arithmetic producing zero does not drop the entry, so a pattern stays stable across
  updates.
- **Aliasing.** A destination may alias an input only where a primitive says so, and those that permit it say
  which buffer. Scatter and gather destinations must not overlap their supports.
- **Mask behavior.** A mask selects positions that remain eligible. A masked reduction over an empty selection
  returns the identity and reports no position rather than position zero.
- **Nonfinite values.** Ordinary primitives propagate NaN and infinity as the arithmetic produces them. The
  checked variants are separate operations: checked scatter and ordered dot reduction detect nonfinite results
  and product underflow and report them, and ordinary vendor BLAS is not a substitute for that contract. Those
  checked workflows, along with pivot eligibility and tolerance policy, move to Klause with their numerical
  behavior intact rather than being weakened into the common surface.

## Consumer baselines

Rechecked for this PR. Klause has moved since the plan recorded it, so its baseline is restated here.

| Repository | Plan recorded | Rechecked at | Change |
|---|---|---|---|
| klause | `87181fd43` | `445d1bd5c` | Moved; findings below still hold |
| kumulant | `31a3802` | `31a3802` | Unchanged |

**klause** calls no dense BLAS operation and no sparse Level 2–3 operation, which confirms the plan's finding at
the newer commit. It reaches Koblas through `SparseMatrix`, `Workspace` (`borrow`, `borrowI32`), the `koblas`
default engine, `SingularMatrix` and `KoblasException`, and through eight `SparseSlices` entry points:
`scatterAxpy` (3 call sites), `clearTouched` (2), and one each of `scatterAxpyChecked`, `reduceDotChecked`,
`pivotCandidatePositions`, `gatherTouched`, `gatherClearTouched` and `activeColumnMaxAbs`. Those eight are the
concrete surface K2 has to move into Klause with its numerical behavior intact; the two checked ones carry the
nonfinite and underflow detection that vendor BLAS cannot provide.

**kumulant** is on a markedly older API: 293 references to `F64VectorLike`, 270 to `F64DenseVector`, 81 to
`F64DenseMatrix`, plus `F64SparseVector`, `F64StridedVectorView`, `F64VectorStorage` and
`F64CholeskyDecomposition`. Those prefixed names no longer exist, so U is a rename-and-rework rather than a
signature adjustment. Its dense usage is small and matches what the plan assigns it: three `cholesky` calls,
two `syrk`, one `gemv`. It uses no panel, tile or packed API, confirming that removing those does not touch it.

**HFactor**, in this repository, calls no dense BLAS operation: a search for every bound dense entry point finds
nothing. It does use `DenseMatrix`, but only as a container for right-hand sides and results in
`SparseFactorization.solve`/`solveInto` and their shape checks, reading `rows` and `cols` and constructing one
through `DenseMatrix.zero`. So the dense vendor surface this PR adds reaches no HFactor call site, and K3's
engine move affects it only through that container, without changing HFactor's algorithms.
