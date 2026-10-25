plugins { kotlin("multiplatform") version "2.4.21" }

repositories {
    maven { url = uri("../../koblas/build/staging-repo") }
    maven("https://central.sonatype.com/repository/maven-snapshots/") {
        mavenContent { snapshotsOnly() }
        content { includeGroup("com.eignex") }
    }
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
            dependencies { implementation("com.eignex:koblas:0.0.0-consumer") }
        }
        commonTest.dependencies { implementation(kotlin("test")) }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
    providers.gradleProperty("iosSimulatorDevice").orNull?.let { device.set(it) }
}
