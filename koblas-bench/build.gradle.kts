import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    kotlin("multiplatform") version "2.3.20"
    kotlin("plugin.allopen") version "2.3.20"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.13"
}

repositories {
    mavenCentral()
}

kotlin {
    applyDefaultHierarchyTemplate()
    // Matches koblas-openblas, whose FFM bindings need 25; this module is dev-only.
    jvmToolchain(25)
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm()

    sourceSets {
        commonMain.dependencies {
            implementation(project(":koblas"))
            implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.13")
        }
        jvmMain.dependencies {
            // Lets the benchmarks compare the discovered OpenBLAS backend against the reference.
            runtimeOnly(project(":koblas-openblas"))
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
        register("level1") { include("Level1Benchmark") }
        register("level2") { include("Level2Benchmark") }
        register("level3") { include("Level3Benchmark") }
        register("solve") { include("SolveBenchmark") }
        register("blockSolve") { include("BlockSolveBenchmark") }
        register("sparse") { include("SparseBenchmark") }
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
