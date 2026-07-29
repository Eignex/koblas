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
        // Block-versus-column solve crossover across right-hand-side counts.
        register("blockSolve") {
            include("BlockSolveBenchmark")
            warmups = 3
            iterations = 5
            iterationTime = 300
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
        }
        // Level-2 sweep out to sizes where bandwidth, not dispatch, decides.
        register("gemv") {
            include("GemvBenchmark")
            warmups = 3
            iterations = 5
            iterationTime = 500
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
        }
        // Level-1 crossover sweep: run once as-is and once with -Pkoblas.noSimd=true.
        register("level1") {
            include("Level1Benchmark")
            warmups = 3
            iterations = 5
            iterationTime = 500
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
        }
        register("probe") {
            include("DenseBenchmark.gemv")
            warmups = 1
            iterations = 1
            iterationTime = 200
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
        }
        named("main") {
            warmups = 3
            iterations = 5
            iterationTime = 1
            iterationTimeUnit = "s"
            advanced("jvmForks", "1")
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
