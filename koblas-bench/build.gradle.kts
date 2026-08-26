import kotlinx.benchmark.gradle.BenchmarkConfiguration
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

/**
 * Registers a suite named [name] over the benchmarks of [className] whose names match [methods].
 *
 * The include pattern is a regex over the fully qualified name, so both ends are anchored: a bare
 * "SolveBenchmark" also matches BlockSolveBenchmark and silently runs two suites as one.
 */
fun NamedDomainObjectContainer<BenchmarkConfiguration>.suite(
    name: String,
    className: String = name.replaceFirstChar(Char::uppercaseChar) + "Benchmark",
    methods: String = "\\w+",
    configure: BenchmarkConfiguration.() -> Unit = {},
) = register(name) {
    include("\\.$className\\.($methods)$")
    configure()
}

plugins {
    kotlin("multiplatform") version "2.4.10"
    kotlin("plugin.allopen") version "2.4.10"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.17"
}

repositories {
    mavenCentral()
}

kotlin {
    applyDefaultHierarchyTemplate()
    // Dev-only module; the incubator vector API the JVM kernels use needs a recent JDK.
    jvmToolchain(25)
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm()
    // Native is where the portable kernels are scalar (no Vector API) and an FFI call is cheap, so the
    // JVM crossovers do not transfer and have to be measured separately. Only the host's own target can
    // run; macOS numbers come from a macOS checkout.
    linuxX64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(project(":koblas"))
            implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.17")
            // koblas keeps serialization compileOnly, which the JVM tolerates but the native klib
            // resolver does not: it needs every transitive klib present to generate the harness.
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        jvmTest.dependencies {
            implementation(kotlin("test-junit"))
        }
    }
}

allOpen {
    annotation("org.openjdk.jmh.annotations.State")
}

benchmark {
    targets {
        register("jvm")
        register("linuxX64")
        register("macosArm64")
    }
    configurations {
        // Settings are shared here rather than repeated per suite: comparing two runs is only meaningful
        // when both spent the same warmup and iteration time.
        configureEach {
            warmups = 3
            iterations = 5
            iterationTime = 500
            iterationTimeUnit = "ms"
            // JMH defaults to several forks; one is enough here and keeps a sweep to a coffee break.
            advanced("jvmForks", "1")
        }

        // One suite per benchmark class, so `benchmark` runs everything and `<name>Benchmark` runs one class.
        suite("level1")
        suite("level2")
        suite("level3")
        suite("solve")
        suite("blockSolve")
        suite("sparse")
        suite("sparseHost")
        suite("basis")
        suite("cholesky")
        suite("qr")
        suite("sparseLevel1")
        suite("matrixOps")
        suite("symbolic")

        // Focused suites for A/B work on a contended machine. Each pins one decisive row plus a control the
        // change under test cannot affect, so a comparison fits a short quiet window instead of needing
        // several undisturbed minutes. The shared settings above still apply, so runs stay comparable.
        suite("qrFocused", "QrBenchmark", "qrPivotedSquare|qrSquare|applyQ") {
            param("n", "512")
            param("backend", "reference")
        }
        suite("level3Focused", "Level3Benchmark", "syr2k|gemm") {
            param("n", "256")
        }
        // The solve rows alone, at the sizes the main suite stops short of. A solve over an existing factor
        // is a pair of triangular solves, and those only cross around 1024, so a sweep ending at 256 cannot
        // see the crossover the level-2 gate on solves turns on. Kept separate because carrying 1024 and
        // 2048 through all ten of the suite's benchmarks and three arms would cost far more than the
        // question needs.
        suite("solveFocused", "SolveBenchmark", "luSolve|luSolveInto|ldlSolveInto") {
            param("n", "256", "1024", "2048")
            param("backend", "reference", "forced-solve")
        }
        suite("sparseFocused", "SparseBenchmark", "sparseLuFactor|sparseGemv") {
            param("n", "256")
        }

        // Gates for a shared-helper change in a hot path. Each pins the rows the change could regress plus
        // at least one control it cannot touch, so drift in the control means the machine moved.
        //
        // JMH runs a suite's rows alphabetically, which decides where a control can sit. A control that sorts
        // after the subjects is worth most, since it sees interference that arrives mid-run: one discarded run
        // had a pristine leading control beside rows three times worse. Where the subject names sort last in
        // their class no trailing control exists, and those gates are kept to a couple of rows instead, so the
        // window is small. Either way the procedure in measure.sh is what makes a comparison sound: several
        // runs a side, compare minimums, read the subject against a control from the same run.
        //
        // Several forks, unlike the single fork above. One fork reports a tight error bar and still moves 25%
        // between two runs of identical code, because the whole run inherits one JVM's JIT decisions. Forks
        // are what make two runs comparable, which is the only thing a gate is for.
        val gate: BenchmarkConfiguration.() -> Unit = {
            warmups = 5
            iterations = 10
            advanced("jvmForks", "5")
        }

        // Bracketed: asumBench leads, iamaxBench trails, the three subjects sit between them.
        suite("scalarKernelsGate", "Level1Benchmark", "asumBench|axpyBench|dot|dot4|iamaxBench") {
            gate()
            param("len", "64", "1024")
            param("kernels", "builtin")
        }
        // scaleBench sorts last in its class, so nrm2 can only lead. Two rows keep the window short.
        suite("scaleGate", "Level1Benchmark", "nrm2|scaleBench") {
            gate()
            param("len", "64", "1024")
            param("kernels", "builtin")
        }
        // Bracketed: sparseGemv leads, sparseTranspose trails, and it is O(nnz) so it costs little.
        suite("sweepGate", "SparseBenchmark", "sparseGemv|sparseLuBtran|sparseLuFtran|sparseTranspose") {
            gate()
            param("n", "256")
        }
        // The sparse level-1 kernels a target does not accelerate, against sparseDotDense, which stays a real
        // override and so carries no forwarding hop. That control sorts third of five, the best this class
        // offers, since it is the only row the change cannot touch.
        val sparseKernelRows = "sparseAsum|sparseAxpy|sparseDotDense|sparseDotSparse|sparseNrm2"
        suite("sparseKernelGate", "SparseLevel1Benchmark", sparseKernelRows) {
            gate()
            param("density", "0.01")
            param("len", "4096")
        }
        // The two gathering kernels against sparseNrm2, which sorts after both and reduces over the stored
        // values alone, so it never touches the dense operand a gather reads. Two lengths at one density,
        // since a gathered load amortizes its setup over its block and a handful of blocks need not behave
        // like several hundred. Answering the pair with `DoubleVector.fromArray` over an index map is slower
        // on every row of an AVX2 host, 2.7x to 8.9x on its efficiency cores and 1.1x to 4.6x on its
        // performance cores, so both kernels stay portable and this suite is what would have to say otherwise.
        suite("sparseGatherGate", "SparseLevel1Benchmark", "sparseGather|sparseGatherZero|sparseNrm2") {
            gate()
            param("density", "0.01")
            param("len", "4096", "65536")
        }
        suite("pivotedQrGate", "PivotedQrGateBenchmark") {
            gate()
        }
        // The triangular rows sort last in Level2Benchmark and Level3Benchmark, so no control can trail them.
        // These stay short, and trsmRightGate is the instrument for the row a shared helper is most likely to
        // cost, the right-hand path that runs one core call per row of B.
        suite("triangularGate", "Level2Benchmark", "gemv|trmv|trsv") {
            gate()
            param("n", "256")
            param("backend", "reference")
        }
        suite("trsmRightGate", "Level3Benchmark", "gemm|trsmRight") {
            gate()
            param("n", "64")
            param("backend", "reference")
        }
        suite("trmmRightGate", "Level3Benchmark", "gemm|trmmRight") {
            gate()
            param("n", "64")
            param("backend", "reference")
        }
        suite("triangularBlockGate", "Level3Benchmark", "gemm|trmm|trsm") {
            gate()
            param("n", "64")
            param("backend", "reference")
        }
    }
}

tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector")
}
// The benchmark runner is a JavaExec whose JVM args the JMH forks inherit; without this the
// reference backend benchmarks silently run the scalar kernels (verify via the setup println).
// -Pkoblas.noSimd=true withholds the module to measure the scalar kernels deliberately.
tasks.withType<JavaExec>().configureEach {
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}

// A soak rather than a benchmark: it looks for a wrong answer under contention, not for a time. Kept out of
// the test suite because catching a regression needs far more rounds than a 300ms test budget allows.
tasks.register<JavaExec>("stressRegistration") {
    group = "verification"
    description = "Hammers concurrent backend registration, failing if a weaker offer ever holds a half."
    val main = kotlin.targets.getByName("jvm").compilations.getByName("main")
    classpath = files(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("com.eignex.koblas.bench.RegistrationStressKt")
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    listOf("rounds", "threads").forEach { name ->
        project.findProperty(name)?.let { args("--$name=$it") }
    }
}
