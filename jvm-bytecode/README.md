# JVM arithmetic boundaries

`./gradlew check` runs `:koblas:verifyKernelBoundaries` against the packaged JVM artifact.
Project calls carrying Vector API values, masks or shuffles must be source-inlined. The verifier
checks typed direct invocations, method handles and lambda bootstrap references without relying
on Kotlin metadata. Primitive/array kernels may remain outlined. The tool is a build dependency
and is not shipped with the library.

`jvmJar` gives expression and cast null assertions inside `internal.jvm25.simd` an explicit local
non-null fast path. Null still reaches the original Kotlin failure helper. Both one- and two-argument
forms retain argument evaluation, failure messages, caller frames and casts. Parameter assertions
and all other namespaces remain byte-for-byte intact. The pass is idempotent; the boundary verifier
rejects unguarded backend assertions. Every output class is verified before replacing the JAR.
The tool classpath and JDK runtime version participate in Gradle cache inputs.

`check` also runs `simdVectorBoundaryAllocationCheck` in normal and narrow/no-FMA JVMs
(unless SIMD is withheld with `koblas.noSimd`). These focused probes forbid HotSpot from inlining
`Intrinsics.checkNotNull`: absolute sum and both maximum-index paths must remain allocation-free
at lengths 17, 129, 4097 and 65537. These probes exercise an opaque-call boundary without
depending on a particular caller's JIT budget.

At arithmetic seams in `internal.kernels`, `internal.scheduling` and `internal.operations`,
the same verifier rejects surviving project calls that accept generic `FunctionN` arguments.
Source-inline these higher-order helpers or use a primitive callback interface when work must
be retained. Startup and allocation-on-miss factories, and generated public facade bridges,
remain ordinary calls. This rule does not require inlining ordinary primitive/array helpers.

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
