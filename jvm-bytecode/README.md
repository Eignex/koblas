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
at lengths 17, 129, 4097 and 65537. The original artifact fails this probe at 512 bytes per call
on the first absolute-sum case; the local-fast-path artifact records zero in every case. This
reproduces an opaque-call failure without depending on a particular caller's JIT budget.

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

An earlier experiment removed internal null assertions broadly. That policy is not included:
its results below explain why ordinary array and parameter checks are retained.

The initial Linux x86_64 experiment used 136 workloads and three paired passes on JDK 25,
with an Intel Xeon Platinum 8488C. [Normal tiering](https://lab.eignex.com/jobs/11925)
showed less than 0.6% improvement in each JVM arm's geometric mean, with workload timing
shifts in both directions. [C1-only and C2-only](https://lab.eignex.com/jobs/11928) showed roughly
2–3% and 2% improvement respectively. These results did not justify adopting the policy. Native source was unchanged; each revision's built executable hash is retained,
and per-case control variation must be considered.

Follow-up comparisons added two arms of the exact same retained revision as an A/A control.
[Shrinking-window scalar GER32](https://lab.eignex.com/jobs/11935) remained slower in all six
paired repetitions, while [other candidates](https://lab.eignex.com/jobs/11937) included both
repeatable slowdowns and shifts that did not reproduce. Identical-arm timings also varied;
the initial aggregate differences do not establish a causal benefit.

The [measured-fork GER logs](https://lab.eignex.com/jobs/11936) show why smaller bytecode can
lose optimization: removing three array parameter checks shrinks `scalarRankUpdate` from
258 to 252 bytecode bytes, but its compiled instruction span grows from 2472 to at least
2656 bytes. C2 then refuses to inline it as already compiled into a big method. Local
ablations recover performance by restoring just this method, or its destination-array
null check; padding removed calls to preserve bytecode offsets does not recover it.
Explicit checks can supply useful nullness facts even when valid callers never pass null.
Blanket assertion removal is therefore not a general recipe for better JIT compilation.

The more general rules are:

- Keep normal tiering as the primary deployment comparison. C1 profiles can help C2 optimize;
  forcing C2 alone changes more than the compiler that eventually runs a hot method.
- Keep outlined numerical kernels on primitive and array signatures. A large primitive-only
  method does not have to inline into its caller to avoid vector allocation.
- Source-inline small helpers that carry Vector API values across a call boundary.
- Source-inline generic numeric callbacks, or use a primitive signature when the callback
  must remain outlined. The benchmark's generic callback can box a `Double` even under C2.
- Inspect the measured fork's compilation log and attribute allocation to its stack before
  changing kernels. Expected scratch allocation and benchmark boxing are separate costs.

[Structured diagnostics](https://lab.eignex.com/jobs/11930) retain 84 measured-fork XML logs
and allocation profiles. C1-only Vector API execution allocated heavily in both policy arms;
removing assertions did not give it C2's vector optimizations. This is a separate issue from
the cost of generated assertions.

A [136-workload source-inlining trial](https://lab.eignex.com/jobs/11948) added `inline` to forty
primitive/array scalar helpers. Normal-tiered geometric mean timing changes were nearly flat (default −1.26%,
scalar −1.16%, SIMD +0.46%, Native −1.47%), with repeatable individual slowdowns as well as wins.
Native source was also affected in this trial. A [selected C1/C2 subset](https://lab.eignex.com/jobs/11952)
looked more favorable, but does not establish a whole-codebase benefit. The
[four-arm assertion interaction](https://lab.eignex.com/jobs/11953) still showed a repeatable
empty sparse-solve slowdown from stripping after source inlining. These results reject blanket
scalar inlining and blanket assertion removal as enforceable performance policies.

The Vector-local fast-path candidate was compared separately in a
[136-workload, three-pass sweep](https://lab.eignex.com/jobs/11962), with 816 accepted captures
and 133 supported workloads per exact engine. Its normal-tiered timing geometric mean was
+1.78% for SIMD and +0.04% for the unchanged scalar control. This is not evidence of a general
throughput improvement. Five sparse cases were over 5% slower in all three passes, despite
unchanged numerical bytecode; absolute sum also had large shifts in both directions.
[Five-case A/A/B confirmation](https://lab.eignex.com/jobs/11966) and a separate
[skewed SpMM confirmation](https://lab.eignex.com/jobs/11967) used two identical original
revision arms and six complete repetitions. No candidate case remained over 5% slower in all
six repetitions against either original arm. The identical-revision SpMM controls themselves
varied by up to 74% in one repetition. These confirmations do not establish throughput
equivalence; they reject treating the initial shifts as established regressions. The intended
benefit is the reproducible removal of Vector-carrier allocation when null-helper inlining fails.

The benchmark module also runs `verifyBenchmarkCallbacks` before preparation and as part of
`check`. Timed `BenchmarkOperation` callbacks expose `invoke()D`. The verifier walks their
harness-side call paths, including nested helpers and virtual dispatch, and rejects erased
`FunctionN.invoke` calls, primitive box/unbox calls, and boxed method-reference adaptation.
It handles invokedynamic and class-based SAM implementations. Cold setup/reporting and the
measured library operation are outside this harness check. Captures and regression controls
record `primitive-double-v1`; compare numerical changes using the same callback representation.

After converting the harness, [job 11968](https://lab.eignex.com/jobs/11968) compared both numerical
policies with the same primitive benchmark bytecode on 13 selected workloads across normal,
C1-only and C2-only compilation. All 234 captures completed. Normal-tiered SIMD timing remained
flat; selected forced-compiler aggregates looked more favorable but retained substantial variation
in unchanged controls. The benchmark README records the results and their limits. The structural
rule removes a specific source of JIT-dependent allocation, rather than promising that every
larger compiled caller runs faster.
