# JVM arithmetic boundaries

`./gradlew check` runs `:koblas:verifyKernelBoundaries` against the packaged JVM artifact.
Project calls carrying Vector API values, masks or shuffles must be source-inlined. The verifier
checks typed direct invocations, method handles and lambda bootstrap references without relying
on Kotlin metadata. Primitive/array kernels may remain outlined. The tool is a build dependency
and is not shipped with the library.

Production JVM compilations reject blanket null-assertion suppression flags. `jvmJar` gives expression and cast
null assertions inside `internal.jvm25.simd` an explicit local
non-null fast path. Null still reaches the original Kotlin failure helper. Both one- and two-argument
forms retain argument evaluation, failure messages, caller frames and casts. Parameter assertions
and all other namespaces remain byte-for-byte intact. The pass is idempotent; the boundary verifier
rejects unguarded backend assertions. Every output class is verified before replacing the JAR.
The tool classpath and JDK runtime version participate in Gradle cache inputs.

`check` also runs the isolated dense and sparse SIMD allocation profiles, each in ordinary,
no-FMA and narrow/no-FMA JVMs, unless SIMD is withheld with `koblas.noSimd`. They cover raw
panels and whole operations with retained scratch. These JVMs run without coverage instrumentation
because instrumentation itself can prevent Vector carrier elimination.

The two focused `simdVectorBoundaryAllocationCheck` profiles additionally forbid HotSpot from
inlining `Intrinsics.checkNotNull`: absolute sum and both maximum-index paths must remain
allocation-free at lengths 17, 129, 4097 and 65537. These probes exercise an opaque-call boundary
without depending on a particular caller's JIT budget. Runtime alignment and vendor checks remain
part of the full target gate.

At arithmetic seams in `internal.kernels`, `internal.scheduling` and `internal.operations`,
the same verifier rejects surviving project calls that accept generic `FunctionN` arguments.
Source-inline these higher-order helpers or use a primitive callback interface when work must
be retained. Startup and allocation-on-miss factories, and generated public facade bridges,
remain ordinary calls. This rule does not require inlining ordinary primitive/array helpers.

Dense and sparse schedules receive primitive providers and operand windows as arrays, offsets, strides
and extents. The typed entry layer validates and unwraps operands once, before scheduling. The packaged
verifier rejects public engine and container fields, arguments and results at scheduling declarations and calls,
including direct constructors, helper results, method handles and dynamic call sites/constants. Public
engine and container implementations are recognized through the packaged inheritance graph. These rules
keep typed container and public engine adaptation outside numerical scheduling. They check visible
bytecode types rather than proving the absence of arbitrary allocation through erased objects or reflection;
the isolated allocation probes independently check the measured paths under C1 and C2. Primitive helpers and
backend dispatch may remain outlined; storage operations and custom public engine adapters remain
outside numerical scheduling.

Reduction adapters likewise pass primitive windows directly to built-in kernels. Custom public engines
retain typed vector calls, since they need a view to receive the original offset, length and stride.
`simdReductionAllocationCheck` keeps vector factories and matrix column-view factories out of line;
`scalarReductionAllocationCheckC1` exercises the same adapters without C2 escape analysis, including
when SIMD is withheld. Scalar-returning reductions must allocate no temporary containers, and axis
sums may allocate only their returned arrays. The adapters themselves remain ordinary outlined calls.

Dense Level 2 vector snapshots also pass primitive windows into the shared schedules. Identical mappings
share one loan; disjoint windows retain their original offsets and strides. Host bindings consume those
windows directly through the same shared preparation. Rectangular
GER checks overlap against primitive destination geometry rather than constructing a matrix view.
`simdAliasStagingAllocationCheck` and `scalarAliasStagingAllocationCheckC1` enforce allocation-free
vector staging with retained scratch while vector and matrix-view factories remain outlined. They cover
aliased GEMV/SYMV, rank updates, shared snapshots and both overlapping and disjoint GER regions.

Dense Level 3 matrix snapshots carry arrays, offsets and leading dimensions through preparation and
scheduling as well. Identical operand regions share one loan, symmetric operands retain their selected
triangle rules, and thin GEMM keeps its GEMV path over the same primitive geometry. Host callbacks consume the same primitive geometry. `simdMatrixAliasAllocationCheck` and
`scalarMatrixAliasAllocationCheckC1` keep the matrix factory outlined while exercising every dense
Level 3 family, padded regions, shared and distinct aliases, thin products and a wider product.

Source inlining removes a typed helper call before C1 or C2 compiles the caller. It does not
guarantee that a larger caller compiles faster or that dispatch inlines. The checker cannot infer
Vector values hidden by erased `Object` signatures. Allocation profiles remain necessary, and
C1 alone does not supply C2's Vector API intrinsics.

The build-time tools use JDK 25's Class-File API. Baseline classes still target Java 17,
and the tools add no runtime dependency. Packaged tests exercise Java 17 and Java 25.

Keep ordinary array and parameter checks. They supply nullness facts to the JIT, and removing
them changes compilation and inlining decisions. Neither blanket assertion removal nor blanket
scalar inlining is an enforceable performance policy.

The more general rules are:

- Keep normal tiering as the primary deployment comparison. C1 profiles can help C2 optimize;
  forcing C2 alone changes more than the compiler that eventually runs a hot method.
- Keep outlined numerical kernels on primitive and array signatures. A large primitive-only
  method does not have to inline into its caller to avoid vector allocation.
- Source-inline small helpers that carry Vector API values across a call boundary.
- Source-inline generic numeric callbacks, or use a primitive signature when the callback
  must remain outlined. Generic callbacks can box a `Double` even under C2.
- Inspect the measured fork's compilation log and attribute allocation to its stack before
  changing kernels. Expected scratch allocation and benchmark boxing are separate costs.

The benchmark module also runs `verifyBenchmarkCallbacks` before preparation and as part of
`check`. Timed `BenchmarkOperation` callbacks expose `invoke()D`. The verifier walks their
harness-side call paths, including nested helpers and virtual dispatch, and rejects erased
`FunctionN.invoke` calls, primitive box/unbox calls, and boxed method-reference adaptation.
It handles invokedynamic and class-based SAM implementations. Cold setup/reporting and the
measured library operation are outside this harness check. Captures and regression controls
record `primitive-double-v1`; compare numerical changes using the same callback representation.
Inherited callback and helper bodies are resolved through their superclass and interface declarations,
including an implementation inherited from a superclass that does not itself implement the callback interface.
Nested lambda targets are followed from timed code, including callbacks constructed during setup and
later invoked through JDK interfaces. Class-based implementations of those interfaces are checked too.
Dispatch conservatively includes every packaged implementation of a reachable interface signature;
it does not infer which callback instance a field or parameter holds. Opaque `MethodHandle.invoke`,
`invokeExact` and `invokeWithArguments` calls are rejected on timed harness paths because their
runtime targets cannot be inspected. External numerical-library bytecode remains outside this check.
Lambda argument and result adaptation and
explicit wrapper constructors count as boxing, while widening between primitive results remains allowed.

Backend implementations are separate from the public typed `KoblasEngine` API. The shared
`BackendEngine` adapter validates and stages operands before passing primitive windows to
the vector, indexed, panel, packed-product and triangular kernel contracts, or `HostBlas`.
`KernelProvider` carries shared identity and capability metadata; `KernelSet` accepts a distinct type
for each arithmetic role. The verifier recognizes each role independently and rejects typed container
arguments/results in those interfaces and their implementations through the packaged inheritance graph,
inheritance of the typed public API, and calls that reconstruct or consume typed containers within a
backend. This also covers unused declarations.
JVM foreign-memory transfers and Native pinning accept the same primitive geometry.
