# JVM arithmetic boundaries and assertion experiment

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

Source inlining removes a typed helper call before C1 or C2 compiles the caller. It does not
guarantee that a larger caller compiles faster or that dispatch inlines. The checker cannot infer
Vector values hidden by erased `Object` signatures. Allocation profiles remain necessary, and
C1 alone does not supply C2's Vector API intrinsics.

Build the experimental packaged JVM artifact with:

```bash
./gradlew check lintDocs -Pkoblas.stripInternalAssertions=true
```

The default build retains null failures. This separate opt-in removal policy runs on `jvmJar`, so publication,
packaged tests and benchmark preparation consume the same transformed artifact. It uses
JDK 25's Class-File API at build time; the baseline classes still target Java 17 and the
tool adds no runtime dependency. The policy and tool classpath participate in the JAR's
Gradle cache inputs.

Inside `com.eignex.koblas.internal`, it removes generated parameter and expression null
assertion calls, trusting Kotlin's declared non-null contracts. It also removes a cast
null assertion immediately after a Vector API `reinterpretShape` result, whose contract
guarantees a non-null vector. Both the stack and local-variable compiler forms are
recognized, with no alternate control-flow entry into the sequence. Unknown producers
retain their assertions. Argument evaluation remains intact.

Public API classes remain byte-for-byte intact. Kotlin metadata, explicit validation,
other `checkNotNull` calls, JVM casts, array bounds checks and numerical instructions
are retained. Kotlin `internal` declarations are public in JVM bytecode; Java callers
of implementation packages must obey the implementation's non-null contracts.

Every output class is verified before the JAR is replaced. The fixture suite checks
public assertions, argument side effects, explicit null failures, vector producer
provenance, alternate entries, wrong-type casts and packaged metadata. Repository tests
exercise the packaged artifact on Java 17 and Java 25. Allocation checks also need the
three SIMD runtime profiles: normal, FMA disabled, and narrow vectors with FMA disabled.

This is a performance experiment. Compare identical numerical sources with the property
disabled and enabled, using the benchmark suite's paired protocol and unchanged Native
arms as controls. A smaller assertion count alone is not performance evidence.

The initial Linux x86_64 experiment used 136 workloads and three paired passes on JDK 25,
with an Intel Xeon Platinum 8488C. [Normal tiering](https://lab.eignex.com/jobs/11925)
showed less than 0.6% improvement in each JVM arm's geometric mean, with workload timing
shifts in both directions. [C1-only and C2-only](https://lab.eignex.com/jobs/11928) showed roughly
2–3% and 2% improvement respectively. These results do not justify enabling the policy
by default. Native source was unchanged; each revision's built executable hash is retained,
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
