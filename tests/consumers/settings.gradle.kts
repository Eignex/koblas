pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositories {
        maven { url = uri("../../koblas/build/staging-repo") }
        maven("https://central.sonatype.com/repository/maven-snapshots/") {
            mavenContent { snapshotsOnly() }
            content { includeGroup("com.eignex") }
        }
        google()
        mavenCentral()
    }
}
rootProject.name = "koblas-consumers"
include(":jvm", ":android")
