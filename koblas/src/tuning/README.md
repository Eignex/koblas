# Dense profile data

Kotlin sources here hold typed performance rules and immutable schedules. Native descriptor legality remains
in the probe catalog. Performance overrides use JVM properties before environment variables through the
existing configuration reader; resolve them once when constructing an engine profile. A malformed override
keeps its conservative default and records a diagnostic. `never`, `always`, and a nonnegative minimum are
distinct choices; zero is permitted and no maximum integer stands for infinity.

The initial block dimensions reproduce the existing `DenseTuning` schedule at source `bb25cc5f`. They are
transitional defaults, not new measured CPU profiles. Packing groups do not determine diagonal or RHS blocks.
The finite native-call work limit is a conservative safety budget for future block consumers, not a calibrated
throughput crossover. New native implementations remain exact-only until a measured rule authorizes AUTO.

An enabled measured entry must identify workload source SHA and case/recipe, hardware, compiler/JDK, native
kernel and layout, comparison paths, report location, and measured range. No startup measurements or generated
cross-language settings are required for these Kotlin-only fields. Each operation migrates its old tuning
readers with its family; the final audit belongs to W27.

AUTO binds a known ordinary variant from explicit preference data, then selects runtime or native arithmetic
per operation. `jvm.c.<operation>.crossover`, `jvm.simd.c.<operation>.crossover`, and
`native.c.<operation>.crossover` are independent rules. Current vector/panel rules count elements; packed
product rules count depth and standalone solves count order. Shape rules and saturating work estimates are
available for the logical block consumers. Incompatible legacy packed geometry always retains the runtime
component, even when an override says `always`. Exact C bypasses crossover rules, retaining capability checks
and semantic early exits. `KoblasEngine.explain` reports the selected operation's component, native ID, width,
and layouts; `tuningDiagnostics` reports invalid overrides. No rule is parsed or plan allocated by a hot call.

The packed and diagonal schedule values now also feed existing `DenseTuning` consumers, so each override is
resolved once. Other family-specific tuning readers remain until their migration or the W27 audit.
