import kotlinx.benchmark.gradle.BenchmarkConfiguration
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

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
    linuxX64()
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
    targets {
        register("jvm")
        register("linuxX64")
        register("macosArm64")
    }
    configurations {
        register("report") {
            warmups = 1
            iterations = 3
            iterationTime = 300
            iterationTimeUnit = "ms"
            advanced("jvmForks", "1")
            include(".*")
            param("n", "256")
            param("len", "4096")
            param("nrhs", "8")
            param("density", "0.01")
            param("shape", "random")
            param("backend", "automatic", "reference")
            param("kernels", "automatic", "scalar")
        }
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
    inputs.dir("src/commonMain")
    outputs.file(layout.buildDirectory.file("checkBenchmarkCoverage/ok.txt"))
    commandLine("python3", "tools/check-benchmark-coverage.py", "benchmark-coverage.tsv")
    doLast { outputs.files.singleFile.apply { parentFile.mkdirs(); writeText("ok") } }
}
tasks.named("check") { dependsOn(checkBenchmarkCoverage) }

tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector")
}
tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}
