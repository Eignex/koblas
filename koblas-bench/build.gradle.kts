import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

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
        // One suite per benchmark class, so `benchmark` runs everything and `<name>Benchmark` runs one
        // class. Settings are shared below rather than repeated per suite: comparing two runs is only
        // meaningful when both spent the same warmup and iteration time.
        configureEach {
            warmups = 3
            iterations = 5
            iterationTime = 500
            iterationTimeUnit = "ms"
            // JMH defaults to several forks; one is enough here and keeps a sweep to a coffee break.
            advanced("jvmForks", "1")
        }
        // Patterns are regexes over the fully qualified name, so they are anchored on the leading dot:
        // a bare "SolveBenchmark" also matches BlockSolveBenchmark and silently runs both suites.
        register("level1") { include("\\.Level1Benchmark\\.") }
        register("level2") { include("\\.Level2Benchmark\\.") }
        register("level3") { include("\\.Level3Benchmark\\.") }
        register("solve") { include("\\.SolveBenchmark\\.") }
        register("blockSolve") { include("\\.BlockSolveBenchmark\\.") }
        register("sparse") { include("\\.SparseBenchmark\\.") }
        register("sparseHost") { include("\\.SparseHostBenchmark\\.") }
        register("cholesky") { include("\\.CholeskyBenchmark\\.") }
        register("symbolic") { include("\\.SymbolicBenchmark\\.") }
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
