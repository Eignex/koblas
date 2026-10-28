plugins { kotlin("multiplatform") version "2.4.21" }

val consumerVersion = providers.gradleProperty("consumer.version").orElse("0.0.0-consumer")

repositories {
    maven { url = uri("../../koblas/build/staging-repo") }
    mavenCentral()
}

kotlin {
    listOf(iosArm64(), iosSimulatorArm64()).forEach {
        it.binaries.framework {
            baseName = "KoblasConsumer"
            isStatic = true
        }
    }
    sourceSets {
        commonMain {
            kotlin.srcDir("../consumers/shared")
            dependencies { implementation("com.eignex:koblas:${consumerVersion.get()}") }
        }
        commonTest.dependencies { implementation(kotlin("test")) }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
    providers.gradleProperty("iosSimulatorDevice").orNull?.let { device.set(it) }
}
