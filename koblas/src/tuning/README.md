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
