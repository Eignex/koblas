# Implementation inventory for issue 666

Baseline: `210a8b2e` (clean working tree). Counts include Kotlin and Java source files, excluding generated build output.

| Module | Production files | Production lines | Test files | Test lines |
| --- | ---: | ---: | ---: | ---: |
| koblas | 91 | 16144 | 118 | 21918 |
| koblas-bench | 26 | 3663 | 18 | 1897 |

The baseline compiles `commonTest` twice, in `test` and `portableTest`. Both `jvmTest` and `jvmTestJarOnJava25` execute the desktop packaged suite. Java 17 executes `portableTest`. `jvmModule` is a descriptor input rather than a source set.

## Baseline configured graph

```text
MODULE :koblas sourceSets=32
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE iosArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosArm64Main/kotlin]
SOURCE iosArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosArm64Test/kotlin]
SOURCE iosMain dependsOn=[appleMain] dirs=[src/iosMain/kotlin]
SOURCE iosSimulatorArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosSimulatorArm64Main/kotlin]
SOURCE iosSimulatorArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosSimulatorArm64Test/kotlin]
SOURCE iosTest dependsOn=[appleTest] dirs=[src/iosTest/kotlin]
SOURCE jvmJava25 dependsOn=[] dirs=[src/jvmJava25/java, src/jvmJava25/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmPortableTest dependsOn=[] dirs=[src/commonTest/kotlin, src/jvmPortableTest/java, src/jvmPortableTest/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain, posixMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest, posixTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
SOURCE posixMain dependsOn=[nativeMain] dirs=[src/posixMain/kotlin]
SOURCE posixTest dependsOn=[nativeTest] dirs=[src/posixTest/kotlin]
COMPILATION java25 sources=[jvmJava25] associates=[main]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION portableTest sources=[jvmPortableTest] associates=[main]
COMPILATION test sources=[commonTest, jvmTest] associates=[java25, main]
MODULE :koblas-bench sourceSets=22
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION test sources=[commonTest, jvmTest] associates=[main]
```

## Source and test sharing

The standard `test` compilation targets Java 17 and includes common conformance, Java interop, packaged-runtime assertions and prepared concurrency tests. `desktopTest` associates with `main`, `test` and `java25` without inheriting their sources. Only `java25` and `desktopTest` receive the Vector API compiler module. Both runtime classpaths use the MR JAR and exclude loose production outputs.

Initial validation: Java 17 packaged baseline ran 691 tests; Java 25 packaged baseline plus desktop ran 858 tests, with no failures. The runtime-aware resource assertion verified root loading on Java 17 and `META-INF/versions/25` loading on Java 25. The old Java 25 packaged task is a dependency entrypoint to `jvmTest`; isolated probes keep separate uninstrumented processes.

Benchmark Linux and Apple directory handling now shares `posixMain`; Windows retains its wide-character implementation. Scalar initialization and benchmark engine identity formatting are shared behind platform selection seams.

## Canonical operands, dispatch and scheduling

All dense matrix operations and host bindings now accept `DenseMatrixRegion`. Dense matrix-vector and rank-update calls accept addressed `DenseVector` operands. Array entrypoints unpack into the same primitive arithmetic cores without allocating convenience views; alias paths retain logical snapshots. Portable panels honor signed vector strides. JVM transfers copy logical regions, selected triangles and stored diagonals at the binding boundary; Native bindings keep pinning and leading dimensions. The obsolete region capability interface, packing adapters and their duplicate test recorders were removed.

Generic products dispatch through the receiver engine, sharing storage classification with operand-based route inspection. Prepared sparse snapshots retain data and a lazy structural transpose, without an engine. Typed and Matrix conveniences use the default engine; benchmark generic/prepared calls name their exact engine explicitly. Triangular conveniences accept the readable Matrix contract, including views and prepared sparse matrices. A custom triangle is normalized without reading its unselected half or implicit diagonal.

Execution and diagnostics share direct product panel enumeration, symmetric product windows, triangular between-block windows, CSC column support, and existing packed block/tile/RHS planning. Benchmark exact-arm admission lives in koblas-bench. RouteKind is shared public vocabulary in the root API package.

Finite-data validation includes the independent textbook/scalar oracles, logical alias snapshots, view padding, signed vector strides, selected triangles, sparse mixed/prepared pairings and benchmark exact-arm attribution.

## Integrated work packages

| Package | Implementation and evidence |
| --- | --- |
| 1. Addressed operands | Every dense operation and vendor matrix contract accepts regions. Signed vector runs reach panels directly; mixed sparse products retain destination origins and leading dimensions. View, alias, triangle, unit-diagonal and full-backing-buffer oracle tests cover the shared implementations. JVM transfers and Native pinning remain distinct. |
| 2. Explicit engine dispatch | Matrix products and their routes share storage dispatch. Prepared operands own immutable data and a lazy structural transpose. Explicit engines govern every pairing; conveniences select the default. Mutation isolation, prepared concurrency and Java consumers exercise this boundary. |
| 3. Shared diagnostics | Direct product panels, symmetric/triangular windows, CSC support and RHS planning are shared by execution and inspection. Operand route overloads derive actual stride, overlap and borrowed facts; recording panels check those decisions. Benchmark admission moved out of production. |
| 4. Capability APIs | Triangular conveniences accept Matrix, including regions/prepared operands and normalized custom inputs. DenseBlas/SparseBlas remain operation surfaces; internal composition owns executors. Serialization identities remain unchanged. API dumps, Java smoke coverage and migration guidance describe deliberate surface changes. |
| 5. Source sharing | One Java 17 baseline test compilation supplies both packaged runtimes. Desktop tests compile separately. POSIX benchmark I/O and scalar initialization are shared; all target leaves remain. |
| 6. Verification wiring | Fourteen explicit process profiles share registration. Java 25 has one canonical Test execution and a retained packaged entrypoint. Packaged classpaths exclude main/java25 loose outputs. Kover includes desktop implementation coverage without counting test classes as production; resolved detekt analysis follows source formatting. |
| 7. Ownership | Major public containers have separate files. Internal dense, sparse, operands, host and diagnostics packages own implementations; Java 25 separates SIMD and host code. Superseded region adapters, staging overloads and test recorders were deleted. |
| 8. Delivery | Normal/full, noSimd, packaged rerun, Native, publication, consumer and targeted benchmark validation are recorded below. Cross-platform runtime results must come from capable CI runners. |

## Resulting configured graph

The library retains 32 source sets; the benchmark has 24 because the POSIX group adds both Main and Test template nodes. Empty template nodes are retained deliberately. Common conformance is no longer scanned into another JVM compilation.

```text
MODULE :koblas sourceSets=32
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE iosArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosArm64Main/kotlin]
SOURCE iosArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosArm64Test/kotlin]
SOURCE iosMain dependsOn=[appleMain] dirs=[src/iosMain/kotlin]
SOURCE iosSimulatorArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosSimulatorArm64Main/kotlin]
SOURCE iosSimulatorArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosSimulatorArm64Test/kotlin]
SOURCE iosTest dependsOn=[appleTest] dirs=[src/iosTest/kotlin]
SOURCE jvmDesktopTest dependsOn=[] dirs=[src/jvmDesktopTest/java, src/jvmDesktopTest/kotlin]
SOURCE jvmJava25 dependsOn=[] dirs=[src/jvmJava25/java, src/jvmJava25/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain, posixMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest, posixTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
SOURCE posixMain dependsOn=[nativeMain] dirs=[src/posixMain/kotlin]
SOURCE posixTest dependsOn=[nativeTest] dirs=[src/posixTest/kotlin]
COMPILATION desktopTest sources=[jvmDesktopTest] associates=[java25, main, test]
COMPILATION java25 sources=[jvmJava25] associates=[main]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION test sources=[commonTest, jvmTest] associates=[main]
MODULE :koblas-bench sourceSets=24
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain, posixMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest, posixTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
SOURCE posixMain dependsOn=[nativeMain] dirs=[src/posixMain/kotlin]
SOURCE posixTest dependsOn=[nativeTest] dirs=[src/posixTest/kotlin]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION test sources=[commonTest, jvmTest] associates=[main]
```

Both packaged runtime classpaths contain the published MR JAR, test outputs and external dependencies. Neither contains loose main or java25 production outputs. Baseline compilation sources are commonTest/jvmTest; desktopTest sources are jvmDesktopTest only, with compiled associations rather than source inheritance.

## Observed source size

Counts include Kotlin/Java under Main/Test source inputs, excluding generated output and the separate module descriptor. They describe organization and API coverage, not a performance or size-reduction claim.

| Module | Production files before → after | Production lines before → after | Test files before → after | Test lines before → after |
| --- | ---: | ---: | ---: | ---: |
| koblas | 91 → 102 | 16144 → 16895 | 118 → 119 | 21918 → 22217 |
| koblas-bench | 26 → 27 | 3663 → 3712 | 18 → 19 | 1897 → 1952 |

## Local validation

`./gradlew --offline check lintDocs` passed. The same JVM suites passed with `-Pkoblas.noSimd=true`, which withholds the Vector API module and leaves installed vendors visible. No crossover/default policy was changed.

| Suite | Tests | Failures/errors/skips |
| --- | ---: | --- |
| Packaged Java 17 baseline | 698 | 0 / 0 / 0 |
| Packaged Java 25 baseline + desktop | 861 | 0 / 0 / 0 |
| JVM benchmark tooling | 103 | 0 / 0 / 0 |
| Linux x64 Native library | 704 | 0 / 0 / 0 |
| Linux x64 Native benchmark tooling | 67 | 0 / 0 / 0 |

A forced `:koblas:jvmTest :koblas:jvmTestJarOnJava25 :koblas:jvmTestJarOnJava17 --rerun-tasks` ran the canonical Java 25 suite once and Java 17 once. The alias executed no second suite. Runtime resource assertions verified JAR-root loading on 17 and META-INF/versions/25 loading on 25. The full check graph also ran each suite once; repeated Gradle output headings from parallel probes are not extra executions.

All fourteen isolated profiles passed: six dense/sparse allocation profiles, narrow-species and no-FMA runtime checks, G1/Serial/ZGC/object-alignment checks, and both oneMKL interface processes. Whole-call dense/structured and retained-workspace checks retain their original zero-allocation thresholds. The consolidation initially exposed convenience-wrapper allocation; primitive addressed cores removed it, without relaxing probes. oneMKL LP64 calls passed under inherited ILP64 configuration; an already initialized ILP64 process was safely rejected.

Refreshed Kover XML includes 22 Java 25 implementation classes, 20 with covered instructions, and no test classes counted as production. Desktop SIMD and FFM binding classes are included while isolated allocation processes remain uninstrumented.

Linux x64 ran locally. Linux arm64, macOS, Windows and iOS sources cross-compiled; Windows test binaries linked. macOS/Windows runtime tests and iOS simulator/device framework publication require capable CI hosts and are not established by cross-compilation.

The staged `0.0.0-consumer-666-final` publication passed all four Java 17/25 classpath/module-path consumers. The artifact verifier found 178 Java 17 base classes and 30 Java 25 desktop classes, valid MR/JPMS metadata, and no desktop APIs/native libraries in Android APKs. The shared fixture checks explicit generic prepared products, mutation isolation, padded backing buffers and signed vector strides. Debug D8 and release R8 instrumentation passed on API 24 and API 34, with one executed consumer test on each device/build-type combination.
