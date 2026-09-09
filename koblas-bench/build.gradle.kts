import kotlinx.benchmark.gradle.BenchmarkConfiguration
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.gradle.jvm.toolchain.JavaLanguageVersion

plugins {
    kotlin("multiplatform") version "2.4.10"
    kotlin("plugin.allopen") version "2.4.10"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.17"
}

repositories { mavenCentral() }

kotlin {
    applyDefaultHierarchyTemplate()
    jvmToolchain(25)
    compilerOptions { freeCompilerArgs.add("-Xexpect-actual-classes") }
    jvm()
    linuxX64 {
        compilations.getByName("main").cinterops.create("benchOpenBlas") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/benchOpenBlas.def"))
            includeDirs(project.file("src/nativeInterop/cinterop"))
        }
        compilations.configureEach {
            if (name.contains("benchmark", ignoreCase = true)) {
                cinterops.create("benchOpenBlasBenchmark") {
                    definitionFile.set(project.file("src/nativeInterop/cinterop/benchOpenBlas.def"))
                    includeDirs(project.file("src/nativeInterop/cinterop"))
                }
            }
        }
    }
    macosArm64()
    sourceSets {
        commonMain.dependencies {
            implementation(project(":koblas"))
            implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.17")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        commonTest.dependencies { implementation(kotlin("test")) }
        jvmTest.dependencies { implementation(kotlin("test-junit")) }
    }
}

allOpen { annotation("org.openjdk.jmh.annotations.State") }

private fun BenchmarkConfiguration.defaults() {
    warmups = 3
    iterations = 5
    iterationTime = 500
    iterationTimeUnit = "ms"
    advanced("jvmForks", "1")
}

benchmark {
    reportsDir = providers.gradleProperty("bench.reportsDir").orElse("reports/benchmarks").get()
    benchsDescriptionDir = providers.gradleProperty("bench.descriptionsDir").orElse("benchsDescription").get()
    targets {
        register("jvm")
        register("linuxX64")
        register("macosArm64")
    }
    configurations {
        fun BenchmarkConfiguration.hardwareDefaults(smoke: Boolean) {
            warmups = 1
            iterations = 1
            iterationTime = if (smoke) 20 else 200
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
            advanced("jmhIgnoreLock", true)

            val comparator = providers.gradleProperty("bench.hardwareComparator").orElse("built-in").get()
            val denseArm = if (comparator == "built-in") "built-in" else comparator
            val sparseArm = if (comparator == "onemkl") "onemkl" else "built-in"
            when (comparator) {
                "built-in" -> include(if (smoke) ".*(?:Level3Benchmark.gemm|SparseProductHostBenchmark.preparedGemv)$" else ".*(?:Level2Benchmark|GemvShapeBenchmark|Level3Benchmark|SyrkBenchmark|Syr2kBenchmark|TrmmBenchmark|TrsmBenchmark|SparseLevel1ComparisonBenchmark|SparseProductHostBenchmark).*")
                "openblas" -> include(if (smoke) ".*Level3Benchmark.gemm$" else ".*(?:Level2Benchmark|GemvShapeBenchmark|Level3Benchmark|SyrkBenchmark|Syr2kBenchmark|TrmmBenchmark|TrsmBenchmark).*")
                "onemkl" -> include(if (smoke) ".*(?:Level3Benchmark.gemm|SparseProductHostBenchmark.preparedGemv)$" else ".*(?:Level2Benchmark|GemvShapeBenchmark|Level3Benchmark|SyrkBenchmark|Syr2kBenchmark|TrmmBenchmark|TrsmBenchmark|SparseLevel1ComparisonBenchmark|SparseProductHostBenchmark).*")
                else -> error("unknown hardware comparator: $comparator")
            }
            param("denseArm", denseArm)
            param("sparseArm", sparseArm)
            if (smoke) {
                param("n", "32")
                param("shape", "16x32")
                param("rankShape", "8x5")
                param("transpose", "false")
                param("lower", "true")
                param("variant", "right-lower-transposed")
            } else {
                param("n", "64", "257")
                param("shape", "16x32", "129x31")
                param("rankShape", "8x5", "129x257")
                param("transpose", "false", "true")
                param("lower", "true", "false")
                param("variant", "right-lower-transposed", "left-upper-transposed")
            }
            param("len", "4096")
            param("density", "0.01")
            param("productShape", "regular")
        }
        register("hardware") { hardwareDefaults(smoke = false) }
        register("hardwareSmoke") { hardwareDefaults(smoke = true) }
        register("full") {
            defaults()
            include(".*")
        }
        register("selected") {
            defaults()
            val requestedInclude = providers.gradleProperty("bench.include").orNull?.takeIf { it.isNotBlank() }
            include(if (requestedInclude != null) "\\.(?:$requestedInclude)$" else "(?!)")
            gradle.startParameter.projectProperties
                .filterKeys { it.startsWith("bench.param.") }
                .forEach { (key, value) ->
                    param(key.removePrefix("bench.param."), *value.split(',').map { it.trim() }.toTypedArray())
                }
        }
    }
}

val checkBenchmarkCoverage = tasks.register<Exec>("checkBenchmarkCoverage") {
    group = "verification"
    description = "Validates the reviewed benchmark coverage manifest."
    inputs.file("benchmark-coverage.tsv")
    inputs.file("comparator-coverage.tsv")
    inputs.file("public-numerical-api.tsv")
    inputs.dir("src/commonMain")
    inputs.dir("../koblas/src/commonMain")
    outputs.file(layout.buildDirectory.file("checkBenchmarkCoverage/ok.txt"))
    commandLine("python3", "tools/check-benchmark-coverage.py", "benchmark-coverage.tsv")
    doLast { outputs.files.singleFile.apply { parentFile.mkdirs(); writeText("ok") } }
}
val testBenchmarkCoverageChecker = tasks.register<Exec>("testBenchmarkCoverageChecker") {
    group = "verification"
    description = "Tests the benchmark coverage checker."
    inputs.file("tools/check-benchmark-coverage.py")
    inputs.dir("tools/test")
    environment("PYTHONDONTWRITEBYTECODE", "1")
    commandLine("python3", "-m", "unittest", "discover", "-s", "tools/test")
}
val benchmarkJvmMetadata = tasks.register("benchmarkJvmMetadata") {
    group = "benchmark"
    description = "Prints metadata for the Java launcher used by JVM benchmarks."
    val launcher = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) }
    doLast {
        val metadata = launcher.get().metadata
        println("executable=${launcher.get().executablePath.asFile.absolutePath}")
        println("version=${metadata.languageVersion}")
        println("vendor=${metadata.vendor}")
        println("runtime=${metadata.jvmVersion}")
    }
}
tasks.named("check") { dependsOn(checkBenchmarkCoverage, testBenchmarkCoverageChecker) }

tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}
tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}
